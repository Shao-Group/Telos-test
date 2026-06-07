"""
Run a full cross-annotation benchmark grid and write a run-index CSV.

Grid axes:
- data_type: sr, cdna, drna, pacbio
- train_annotation: refseq, gencode, ensembl
- test_annotation: refseq, gencode, ensembl (excluding same-as-train by default)

Training is shared per ``(data_type, train_annotation)`` under
``<outdir>/_cross_annotation_shared_train/<dt>__train_<anno>/``.

Parallelism (optional):
- **Shared-train phase**: unique train axes in parallel (``max_parallel_trains``).
- **Cell phase**: matrix cells (e.g. train=gencode test=refseq) in parallel (``max_parallel_cells``).
- **Within cell**: test rows in parallel (``max_parallel_tests``, auto-capped via ``total_cpus``).
"""

from __future__ import annotations

import csv
from pathlib import Path

from telos_v2.benchmark.grid_parallel import (
    BenchmarkCellJob,
    apply_execution_options,
    cap_parallel_tests,
    probe_test_annotation,
    run_benchmark_cells_parallel,
    run_shared_trains_parallel,
    unique_shared_train_jobs,
)
from telos_v2.benchmark.matrix import (
    ANNOTATION_TO_REF_ID,
    DATA_TYPE_TO_MODALITY,
    benchmark_mapping_to_yaml_text,
    build_benchmark_yaml_mapping,
    resolve_bundles_root,
)
from telos_v2.benchmark.orchestrator import run_benchmark
from telos_v2.benchmark.shared_train import apply_train_reuse, shared_train_dir
from telos_v2.config_loader import default_stage1_config_path
from telos_v2.config_models import BenchmarkIO
from telos_v2.config_validation import validate_benchmark_config

SHARED_TRAIN_SUBDIR = "_cross_annotation_shared_train"


def _build_cell_jobs(
    *,
    outdir: Path,
    root: Path,
    stage1: Path,
    dtypes: tuple[str, ...],
    anns: tuple[str, ...],
    include_same_annotation: bool,
    only_same_annotation: bool,
    shared_train_subdir: str,
    max_parallel_tests: int,
    save_pr_tables: bool,
) -> list[BenchmarkCellJob]:
    jobs: list[BenchmarkCellJob] = []
    for dt in dtypes:
        for tr in anns:
            for te in anns:
                if only_same_annotation:
                    if tr != te:
                        continue
                elif not include_same_annotation and tr == te:
                    continue
                combo_id = f"{dt}__train_{tr}__test_{te}"
                jobs.append(
                    BenchmarkCellJob(
                        combo_id=combo_id,
                        data_type=dt,
                        train_annotation=tr,
                        test_annotation=te,
                        combo_outdir=str((outdir / combo_id).resolve()),
                        grid_outdir=str(outdir),
                        shared_train_subdir=shared_train_subdir,
                        bundles_root=str(root),
                        stage1_config=str(stage1),
                        max_parallel_tests=max_parallel_tests,
                        save_pr_tables=save_pr_tables,
                    )
                )
    return jobs


def run_cross_annotation_benchmarks(
    *,
    outdir: Path,
    bundles_root: Path | None = None,
    stage1_config: Path | None = None,
    data_types: tuple[str, ...] | None = None,
    annotations: tuple[str, ...] | None = None,
    include_same_annotation: bool = False,
    only_same_annotation: bool = False,
    shared_train_subdir: str = SHARED_TRAIN_SUBDIR,
    max_parallel_tests: int = 1,
    max_parallel_cells: int = 1,
    max_parallel_trains: int = 1,
    total_cpus: int | None = None,
    save_pr_tables: bool = True,
) -> int:
    """
    Execute all matrix benchmarks for the selected cross-annotation grid.

    Writes a compact run index at ``<outdir>/reports/cross_annotation_runs.csv``.
    Returns ``0`` only if every combination exits successfully.
    """
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir = outdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_bundles_root(bundles_root)
    stage1 = stage1_config if stage1_config is not None else default_stage1_config_path()
    if not stage1.is_file():
        print(f"[telos_v2] cross-annotation: stage1 config not found: {stage1}")
        return 2

    dtypes = data_types if data_types is not None else tuple(DATA_TYPE_TO_MODALITY.keys())
    anns = annotations if annotations is not None else tuple(ANNOTATION_TO_REF_ID.keys())
    if max_parallel_tests < 1:
        max_parallel_tests = 1
    if max_parallel_cells < 1:
        max_parallel_cells = 1
    if max_parallel_trains < 1:
        max_parallel_trains = 1

    effective_tests = cap_parallel_tests(
        max_parallel_cells=max_parallel_cells,
        max_parallel_tests=max_parallel_tests,
        total_cpus=total_cpus,
    )

    cell_jobs = _build_cell_jobs(
        outdir=outdir,
        root=root,
        stage1=stage1,
        dtypes=dtypes,
        anns=anns,
        include_same_annotation=include_same_annotation or only_same_annotation,
        only_same_annotation=only_same_annotation,
        shared_train_subdir=shared_train_subdir,
        max_parallel_tests=effective_tests,
        save_pr_tables=save_pr_tables,
    )
    if not cell_jobs:
        print("[telos_v2] cross-annotation: no matrix cells to run.")
        return 2
    if only_same_annotation:
        print(
            f"[telos_v2] cross-annotation: running {len(cell_jobs)} same-annotation cells only "
            f"(train_annotation == test_annotation).",
            flush=True,
        )

    probe_by_train: dict[tuple[str, str], str] = {}
    for dt in dtypes:
        for tr in anns:
            probe_by_train[(dt, tr)] = probe_test_annotation(
                tr,
                anns,
                include_same_annotation=include_same_annotation or only_same_annotation,
            )

    use_parallel_grid = max_parallel_cells > 1 or max_parallel_trains > 1
    if use_parallel_grid:
        train_jobs = unique_shared_train_jobs(
            cell_jobs,
            probe_by_train=probe_by_train,
            grid_outdir=outdir,
            shared_train_subdir=SHARED_TRAIN_SUBDIR,
            bundles_root=root,
            stage1_config=stage1,
        )
        train_results = run_shared_trains_parallel(
            train_jobs, max_workers=max_parallel_trains
        )
        bad_train = [k for k, (c, _) in train_results.items() if c != 0]
        if bad_train:
            print(f"[telos_v2] cross-annotation: shared-train failed for: {', '.join(bad_train)}")
        if max_parallel_cells > 1:
            rows = run_benchmark_cells_parallel(cell_jobs, max_workers=max_parallel_cells)
        else:
            rows = _run_cells_sequential(cell_jobs, root=root, stage1=stage1)
    else:
        rows = _run_cells_sequential(cell_jobs, root=root, stage1=stage1)

    all_ok = all(r.get("status") == "ok" for r in rows)
    run_index = reports_dir / "cross_annotation_runs.csv"
    _write_run_index(run_index, rows, merge_existing=only_same_annotation)
    print(f"[telos_v2] cross-annotation index: {run_index}")
    return 0 if all_ok else 1


def _write_run_index(
    run_index: Path,
    new_rows: list[dict[str, str]],
    *,
    merge_existing: bool,
) -> None:
    fieldnames = [
        "run_id",
        "data_type",
        "train_annotation",
        "test_annotation",
        "exit_code",
        "status",
        "error",
        "outdir",
        "summary_csv",
        "benchmark_yaml",
    ]
    merged: dict[str, dict[str, str]] = {}
    if merge_existing and run_index.is_file():
        with run_index.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rid = str(row.get("run_id", "")).strip()
                if rid:
                    merged[rid] = {k: str(row.get(k, "")) for k in fieldnames}
    for row in new_rows:
        merged[row["run_id"]] = {k: str(row.get(k, "")) for k in fieldnames}
    run_index.parent.mkdir(parents=True, exist_ok=True)
    with run_index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged[k] for k in sorted(merged))


def _run_cells_sequential(
    cell_jobs: list[BenchmarkCellJob],
    *,
    root: Path,
    stage1: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    n = len(cell_jobs)
    for i, job in enumerate(cell_jobs, start=1):
        print(f"[telos_v2] cross-annotation {i}/{n}: {job.combo_id}", flush=True)
        combo_outdir = Path(job.combo_outdir)
        shared_parent = shared_train_dir(
            Path(job.grid_outdir),
            job.data_type,
            job.train_annotation,
            subdir=job.shared_train_subdir,
        )
        try:
            mapping = build_benchmark_yaml_mapping(
                data_type=job.data_type,
                train_annotation=job.train_annotation,
                test_annotation=job.test_annotation,
                bundles_root=root,
                stage1_config=stage1,
                train_outdir=shared_parent,
            )
            apply_train_reuse(mapping, shared_train_parent=shared_parent)
            apply_execution_options(
                mapping,
                max_parallel_tests=job.max_parallel_tests,
                save_pr_tables=job.save_pr_tables,
            )
            validate_benchmark_config(mapping)
        except (FileNotFoundError, ValueError, OSError) as exc:
            rows.append(
                {
                    "run_id": job.combo_id,
                    "data_type": job.data_type,
                    "train_annotation": job.train_annotation,
                    "test_annotation": job.test_annotation,
                    "exit_code": "2",
                    "status": "failed",
                    "error": str(exc),
                    "outdir": str(combo_outdir),
                    "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                    "benchmark_yaml": "",
                }
            )
            print(f"[telos_v2] cross-annotation setup failed: {exc}")
            continue

        combo_reports = combo_outdir / "reports"
        combo_reports.mkdir(parents=True, exist_ok=True)
        cfg_path = combo_reports / "generated_benchmark.yaml"
        header = (
            "# Cross-annotation benchmark (standard ref_gtf / tmap from bundle manifests).\n"
            f"# Train reuse: {SHARED_TRAIN_SUBDIR}/<dt>__train_<anno>/\n\n"
        )
        cfg_path.write_text(header + benchmark_mapping_to_yaml_text(mapping), encoding="utf-8")
        code = run_benchmark(BenchmarkIO(config=cfg_path, outdir=combo_outdir))
        rows.append(
            {
                "run_id": job.combo_id,
                "data_type": job.data_type,
                "train_annotation": job.train_annotation,
                "test_annotation": job.test_annotation,
                "exit_code": str(code),
                "status": "ok" if code == 0 else "failed",
                "error": "",
                "outdir": str(combo_outdir),
                "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                "benchmark_yaml": str(cfg_path),
            }
        )
    return rows


__all__ = ["SHARED_TRAIN_SUBDIR", "run_cross_annotation_benchmarks"]
