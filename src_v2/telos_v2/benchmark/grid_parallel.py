"""
Process-pool execution for benchmark grids (shared train + parallel matrix cells).

Phase 1 — train each unique ``(data_type, train_annotation)`` once (parallel, file-locked).
Phase 2 — run benchmark cells (e.g. gencode→refseq vs ensembl→gencode) in parallel with ``train.mode=skip``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telos_v2.benchmark.matrix import build_benchmark_yaml_mapping
from telos_v2.benchmark.shared_train import (
    apply_train_reuse,
    shared_models_ready,
    shared_train_dir,
    shared_train_lock,
)
from telos_v2.benchmark.util import as_path
from telos_v2.commands.train import run_train
from telos_v2.config_models import TrainIO


@dataclass(frozen=True)
class SharedTrainJob:
    data_type: str
    train_annotation: str
    probe_test_annotation: str
    grid_outdir: str
    shared_train_subdir: str
    bundles_root: str
    stage1_config: str


@dataclass(frozen=True)
class BenchmarkCellJob:
    combo_id: str
    data_type: str
    train_annotation: str
    test_annotation: str
    combo_outdir: str
    grid_outdir: str
    shared_train_subdir: str
    bundles_root: str
    stage1_config: str
    max_parallel_tests: int
    save_pr_tables: bool


def cap_parallel_tests(
    *,
    max_parallel_cells: int,
    max_parallel_tests: int,
    total_cpus: int | None = None,
) -> int:
    """Limit per-cell test parallelism so ``cells × tests`` does not oversubscribe CPUs."""
    if max_parallel_cells <= 1:
        return max(1, max_parallel_tests)
    cpus = total_cpus if total_cpus is not None else (os.cpu_count() or 1)
    cpus = max(1, int(cpus))
    per_cell = max(1, cpus // max(1, max_parallel_cells))
    capped = min(max(1, max_parallel_tests), per_cell)
    if capped < max_parallel_tests:
        print(
            f"[telos_v2] capped max_parallel_tests {max_parallel_tests} -> {capped} "
            f"({max_parallel_cells} cells × {capped} ≤ ~{cpus} cpus)",
            flush=True,
        )
    return capped


def probe_test_annotation(
    train_annotation: str,
    annotations: tuple[str, ...],
    *,
    include_same_annotation: bool,
) -> str:
    if include_same_annotation:
        return train_annotation
    for te in annotations:
        if te != train_annotation:
            return te
    return annotations[0]


def unique_shared_train_jobs(
    cell_jobs: list[BenchmarkCellJob],
    *,
    probe_by_train: dict[tuple[str, str], str],
    grid_outdir: Path,
    shared_train_subdir: str,
    bundles_root: Path,
    stage1_config: Path,
) -> list[SharedTrainJob]:
    seen: set[tuple[str, str]] = set()
    out: list[SharedTrainJob] = []
    for job in cell_jobs:
        key = (job.data_type, job.train_annotation)
        if key in seen:
            continue
        seen.add(key)
        probe = probe_by_train.get(key)
        if not probe:
            probe = job.test_annotation
        out.append(
            SharedTrainJob(
                data_type=job.data_type,
                train_annotation=job.train_annotation,
                probe_test_annotation=probe,
                grid_outdir=str(grid_outdir),
                shared_train_subdir=shared_train_subdir,
                bundles_root=str(bundles_root),
                stage1_config=str(stage1_config),
            )
        )
    return out


def _worker_ensure_shared_train(job: SharedTrainJob) -> tuple[str, int, str]:
    axis_id = f"{job.data_type}__train_{job.train_annotation}"
    try:
        grid_outdir = Path(job.grid_outdir)
        bundles_root = Path(job.bundles_root)
        stage1_config = Path(job.stage1_config)
        shared_parent = shared_train_dir(
            grid_outdir,
            job.data_type,
            job.train_annotation,
            subdir=job.shared_train_subdir,
        )
        model_dir = shared_parent / "models"
        if shared_models_ready(model_dir):
            return axis_id, 0, ""
        with shared_train_lock(shared_parent / ".train.lock"):
            if shared_models_ready(model_dir):
                return axis_id, 0, ""
            mapping = build_benchmark_yaml_mapping(
                data_type=job.data_type,
                train_annotation=job.train_annotation,
                test_annotation=job.probe_test_annotation,
                bundles_root=bundles_root,
                stage1_config=stage1_config,
                train_outdir=shared_parent,
            )
            train_obj = mapping["train"]
            train_io = TrainIO(
                bam=as_path(train_obj.get("bam"), "train.bam"),
                gtf=as_path(train_obj.get("gtf"), "train.gtf"),
                ref_gtf=as_path(train_obj.get("ref_gtf"), "train.ref_gtf"),
                tmap=as_path(train_obj.get("tmap"), "train.tmap"),
                gtf_pool=tuple(Path(str(p)) for p in (train_obj.get("gtf_pool") or [])) or None,
                tmap_pool=tuple(Path(str(p)) for p in (train_obj.get("tmap_pool") or [])) or None,
                outdir=shared_parent,
                config_file=stage1_config,
            )
            code = run_train(train_io)
            if code != 0:
                return axis_id, code, f"run_train exited {code}"
            if not shared_models_ready(model_dir):
                return axis_id, 1, "models missing after run_train"
            return axis_id, 0, ""
    except (OSError, ValueError, FileNotFoundError) as exc:
        return axis_id, 2, str(exc)


def run_shared_trains_parallel(
    jobs: list[SharedTrainJob],
    *,
    max_workers: int,
) -> dict[str, tuple[int, str]]:
    if not jobs:
        return {}
    workers = max(1, min(max_workers, len(jobs)))
    print(f"[telos_v2] shared-train phase: {len(jobs)} axes, max_workers={workers}", flush=True)
    results: dict[str, tuple[int, str]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_ensure_shared_train, j): j for j in jobs}
        for fut in as_completed(futures):
            axis_id, code, err = fut.result()
            results[axis_id] = (code, err)
            status = "ok" if code == 0 else "failed"
            print(f"[telos_v2] shared-train {axis_id}: {status} (code={code})", flush=True)
    return results


def apply_execution_options(
    mapping: dict[str, Any],
    *,
    max_parallel_tests: int,
    save_pr_tables: bool,
) -> None:
    execution = mapping.setdefault("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("mapping.execution must be a dict")
    execution["max_parallel_tests"] = int(max_parallel_tests)
    analysis = mapping.get("analysis")
    if not isinstance(analysis, dict):
        return
    pr = analysis.get("pr_vs_baseline")
    if isinstance(pr, dict):
        pr["save_pr_tables"] = bool(save_pr_tables)


def _worker_run_benchmark_cell(job: BenchmarkCellJob) -> dict[str, str]:
    from telos_v2.benchmark.matrix import benchmark_mapping_to_yaml_text
    from telos_v2.benchmark.orchestrator import run_benchmark
    from telos_v2.config_models import BenchmarkIO
    from telos_v2.config_validation import validate_benchmark_config

    row: dict[str, str] = {
        "run_id": job.combo_id,
        "data_type": job.data_type,
        "train_annotation": job.train_annotation,
        "test_annotation": job.test_annotation,
        "outdir": job.combo_outdir,
        "summary_csv": str(Path(job.combo_outdir) / "reports" / "benchmark_summary.csv"),
        "benchmark_yaml": "",
        "exit_code": "2",
        "status": "failed",
        "error": "",
    }
    try:
        grid_outdir = Path(job.grid_outdir)
        combo_outdir = Path(job.combo_outdir)
        shared_parent = shared_train_dir(
            grid_outdir,
            job.data_type,
            job.train_annotation,
            subdir=job.shared_train_subdir,
        )
        mapping = build_benchmark_yaml_mapping(
            data_type=job.data_type,
            train_annotation=job.train_annotation,
            test_annotation=job.test_annotation,
            bundles_root=Path(job.bundles_root),
            stage1_config=Path(job.stage1_config),
            train_outdir=shared_parent,
        )
        apply_train_reuse(mapping, shared_train_parent=shared_parent)
        train_block = mapping.get("train")
        if isinstance(train_block, dict) and str(train_block.get("mode", "")).lower() != "skip":
            row["error"] = (
                "shared models not ready (train.mode=run); run shared-train phase first or "
                f"check {shared_parent / 'models'}"
            )
            return row
        apply_execution_options(
            mapping,
            max_parallel_tests=job.max_parallel_tests,
            save_pr_tables=job.save_pr_tables,
        )
        validate_benchmark_config(mapping)
        combo_reports = combo_outdir / "reports"
        combo_reports.mkdir(parents=True, exist_ok=True)
        cfg_path = combo_reports / "generated_benchmark.yaml"
        header = (
            "# Cross-annotation benchmark (parallel grid cell).\n"
            f"# shared_train={job.shared_train_subdir}/\n\n"
        )
        cfg_path.write_text(header + benchmark_mapping_to_yaml_text(mapping), encoding="utf-8")
        row["benchmark_yaml"] = str(cfg_path)
        code = run_benchmark(BenchmarkIO(config=cfg_path, outdir=combo_outdir))
        row["exit_code"] = str(code)
        row["status"] = "ok" if code == 0 else "failed"
        return row
    except (OSError, ValueError, FileNotFoundError) as exc:
        row["error"] = str(exc)
        return row


def run_benchmark_cells_parallel(
    jobs: list[BenchmarkCellJob],
    *,
    max_workers: int,
) -> list[dict[str, str]]:
    if not jobs:
        return []
    workers = max(1, min(max_workers, len(jobs)))
    print(f"[telos_v2] benchmark-cell phase: {len(jobs)} cells, max_workers={workers}", flush=True)
    rows: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_run_benchmark_cell, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            job = futures[fut]
            row = fut.result()
            rows.append(row)
            print(
                f"[telos_v2] cell {done}/{len(jobs)}: {job.combo_id} -> {row['status']} (code={row['exit_code']})",
                flush=True,
            )
    return rows


__all__ = [
    "BenchmarkCellJob",
    "SharedTrainJob",
    "apply_execution_options",
    "cap_parallel_tests",
    "probe_test_annotation",
    "run_benchmark_cells_parallel",
    "run_shared_trains_parallel",
    "unique_shared_train_jobs",
]
