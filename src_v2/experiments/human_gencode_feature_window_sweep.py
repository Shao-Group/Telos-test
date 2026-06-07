"""
Train and benchmark human GENCODE models across Stage I feature window sizes.

For each ``window_size`` in ``stage1.feature_extraction``, materializes a Stage I YAML,
trains on the canonical bundle sample, and evaluates on all other samples under
``data/bundles/GRCh38_gencode49`` (train and test both use GENCODE v49).

Stage I feature disk cache is disabled (``cache_dir: null`` and ``TELOS_STAGE1_CACHE_DIR``
unset for each benchmark) so every run recomputes features for the active window size.

Parallelism (defaults):
- Up to 4 data types at once via subprocess (``--parallel-data-types``).
- Up to 3 benchmark tests at once per cell (``execution.max_parallel_tests`` in YAML).

Usage::

    PYTHONPATH=src_v2 python src_v2/experiments/human_gencode_feature_window_sweep.py

    PYTHONPATH=src_v2 python src_v2/experiments/human_gencode_feature_window_sweep.py \\
      --outdir runs/human_gencode_feature_window \\
      --window-sizes 50 100 200 \\
      --data-types sr pacbio \\
      --max-parallel-tests 3

Plot results (after runs finish)::

    PYTHONPATH=src_v2 python src_v2/experiments/plot_human_gencode_feature_window_results.py \\
      --root runs/human_gencode_feature_window
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from telos_v2.benchmark.matrix import (
    DATA_TYPE_TO_MODALITY,
    benchmark_mapping_to_yaml_text,
    build_benchmark_yaml_mapping,
    resolve_bundles_root,
)
from telos_v2.benchmark.orchestrator import run_benchmark
from telos_v2.benchmark.timing import append_timing_row, elapsed_s, log_ts, utc_now_iso
from telos_v2.config_loader import default_stage1_config_path, load_mapping_config
from telos_v2.config_models import BenchmarkIO
from telos_v2.config_validation import validate_benchmark_config

ANNOTATION = "gencode"
DEFAULT_WINDOW_SIZES = (25, 50, 75, 100, 125, 150, 175, 200)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_analysis_block() -> dict[str, Any]:
    return {
        "enabled": True,
        "benchmark_mode": "minimal",
        "debug": {"keep_pr_work": False},
        "pr_vs_baseline": {
            "enabled": True,
            "measure": "cov",
            "plot": True,
            "filter_validation_chroms": True,
            "chromosomes_file": None,
            "save_pr_tables": True,
            "gffcompare_bin": "/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare",
        },
    }


def compute_stage1_n_workers_cap(
    *,
    parallel_data_types: int,
    max_parallel_tests: int,
) -> int:
    """Limit Stage I pool size when many benchmark cells/tests run concurrently."""
    nc = os.cpu_count() or 8
    denom = max(1, parallel_data_types) * max(1, max_parallel_tests)
    return max(1, nc // denom)


@contextmanager
def _no_stage1_feature_cache() -> Iterator[None]:
    """Ensure Stage I does not read/write disk cache (``TELOS_STAGE1_CACHE_DIR`` overrides YAML)."""
    key = "TELOS_STAGE1_CACHE_DIR"
    saved = os.environ.pop(key, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[key] = saved


def materialize_stage1_config(
    *,
    base_config: Path,
    window_size: int,
    out_path: Path,
    n_workers: int | None = None,
) -> Path:
    """Write Stage I YAML for this sweep: window sizes scaled; ``cache_dir`` disabled."""
    import yaml  # type: ignore

    cfg = copy.deepcopy(load_mapping_config(base_config.resolve()))
    stage1 = cfg.setdefault("stage1", {})
    if not isinstance(stage1, dict):
        raise ValueError("stage1 block must be a mapping")
    fe = stage1.setdefault("feature_extraction", {})
    if not isinstance(fe, dict):
        raise ValueError("stage1.feature_extraction must be a mapping")
    fe["window_size"] = int(window_size)
    fe["density_window"] = int(window_size)
    fe["coverage_window"] = int(window_size)
    fe["gradient_analysis_range"] = int(window_size)
    fe["cache_dir"] = None
    if n_workers is not None:
        fe["n_workers"] = int(n_workers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    return out_path.resolve()


def run_id_for(*, data_type: str, window_size: int) -> str:
    return f"{data_type}__ws{window_size}__train_{ANNOTATION}__test_{ANNOTATION}"


def run_benchmark_cell(
    *,
    outdir: Path,
    bundles_root: Path,
    base_stage1_config: Path,
    data_type: str,
    window_size: int,
    max_parallel_tests: int,
    parallel_data_types: int,
) -> dict[str, str]:
    """Run one (window_size, data_type) benchmark cell; return index row fields."""
    if data_type not in DATA_TYPE_TO_MODALITY:
        raise ValueError(f"Unknown data_type {data_type!r}")

    rid = run_id_for(data_type=data_type, window_size=window_size)
    combo_outdir = outdir / rid
    combo_reports = combo_outdir / "reports"
    combo_reports.mkdir(parents=True, exist_ok=True)
    config_dir = outdir / "reports" / "stage1_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    sweep_timing = outdir / "reports" / "sweep_timing.csv"

    n_workers_cap = compute_stage1_n_workers_cap(
        parallel_data_types=parallel_data_types,
        max_parallel_tests=max_parallel_tests,
    )
    stage1_path = materialize_stage1_config(
        base_config=base_stage1_config,
        window_size=window_size,
        out_path=config_dir / f"stage1.window_size_{window_size}.yaml",
        n_workers=n_workers_cap,
    )

    row_base = {
        "run_id": rid,
        "data_type": data_type,
        "window_size": str(window_size),
        "outdir": str(combo_outdir),
        "summary_csv": str(combo_reports / "benchmark_summary.csv"),
        "stage1_config": str(stage1_path),
        "stage1_n_workers": str(n_workers_cap),
        "max_parallel_tests": str(max_parallel_tests),
    }

    cell_t0 = time.perf_counter()
    log_ts(f"[sweep] cell start {rid} n_workers={n_workers_cap} max_parallel_tests={max_parallel_tests}")

    try:
        mapping = build_benchmark_yaml_mapping(
            data_type=data_type,
            train_annotation=ANNOTATION,
            test_annotation=ANNOTATION,
            bundles_root=bundles_root,
            stage1_config=stage1_path,
            train_outdir=combo_outdir / "train",
        )
        mapping["generated_by"] = "experiments.human_gencode_feature_window_sweep"
        mapping["analysis"] = _default_analysis_block()
        mapping["execution"] = {
            "stop_on_error": False,
            "max_parallel_tests": int(max_parallel_tests),
        }
        validate_benchmark_config(mapping)
    except (FileNotFoundError, ValueError, OSError) as exc:
        elapsed = elapsed_s(cell_t0)
        log_ts(f"[sweep] cell setup failed {rid} elapsed_s={elapsed}: {exc}")
        append_timing_row(
            sweep_timing,
            {
                **row_base,
                "phase": "cell",
                "status": "setup_failed",
                "elapsed_s": str(elapsed),
                "ended_at": utc_now_iso(),
                "error": str(exc),
            },
        )
        return {
            **row_base,
            "exit_code": "2",
            "status": "setup_failed",
            "error": str(exc),
            "benchmark_yaml": "",
            "elapsed_s": str(elapsed),
            "started_at": utc_now_iso(),
        }

    cfg_path = combo_reports / "generated_benchmark_gencode_window_sweep.yaml"
    header = (
        "# Human GENCODE train+test; feature window size sweep.\n"
        f"# window_size={window_size}\n"
        f"# bundles_root={bundles_root}\n"
        f"# stage1_config={stage1_path} (cache_dir=null; no TELOS_STAGE1_CACHE_DIR)\n"
        f"# max_parallel_tests={max_parallel_tests}\n\n"
    )
    cfg_path.write_text(header + benchmark_mapping_to_yaml_text(mapping), encoding="utf-8")

    with _no_stage1_feature_cache():
        code = run_benchmark(BenchmarkIO(config=cfg_path, outdir=combo_outdir))

    elapsed = elapsed_s(cell_t0)
    ok = code == 0
    log_ts(f"[sweep] cell end {rid} exit_code={code} elapsed_s={elapsed}")
    append_timing_row(
        sweep_timing,
        {
            **row_base,
            "phase": "cell",
            "status": "ok" if ok else "failed",
            "elapsed_s": str(elapsed),
            "ended_at": utc_now_iso(),
            "exit_code": str(code),
        },
    )
    return {
        **row_base,
        "exit_code": str(code),
        "status": "ok" if ok else "failed",
        "error": "",
        "benchmark_yaml": str(cfg_path),
        "elapsed_s": str(elapsed),
        "started_at": utc_now_iso(),
    }


def _worker_main(args: argparse.Namespace) -> int:
    """Subprocess entry: run a single (window_size, data_type) cell."""
    if args.bundles_root is None or args.window_size is None or args.data_type is None:
        log_ts("ERROR: --worker requires --bundles-root, --window-size, --data-type", file=sys.stderr)
        return 2
    parallel_dt = (
        int(args.parallel_data_types_count)
        if args.parallel_data_types_count is not None
        else 1
    )
    base = args.stage1_config if args.stage1_config is not None else default_stage1_config_path()
    if not base.is_file():
        print(f"ERROR: stage1 config not found: {base}", file=sys.stderr)
        return 2
    try:
        bundles_root = resolve_bundles_root(args.bundles_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    row = run_benchmark_cell(
        outdir=args.outdir.resolve(),
        bundles_root=bundles_root,
        base_stage1_config=base,
        data_type=str(args.data_type).strip().lower(),
        window_size=int(args.window_size),
        max_parallel_tests=int(args.max_parallel_tests),
        parallel_data_types=max(1, parallel_dt),
    )
    result_path = (args.outdir / "reports" / f"worker_result__{row['run_id']}.json").resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return int(row["exit_code"])


def run_sweep(
    *,
    outdir: Path,
    bundles_root: Path,
    base_stage1_config: Path,
    data_types: tuple[str, ...],
    window_sizes: tuple[int, ...],
    max_parallel_tests: int,
    parallel_data_types: bool,
    max_parallel_data_types: int,
    log_dir: Path | None,
) -> int:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    logs = (log_dir if log_dir is not None else reports / "worker_logs").resolve()
    logs.mkdir(parents=True, exist_ok=True)

    gencode_bundle = bundles_root / "GRCh38_gencode49"
    if not gencode_bundle.is_dir():
        log_ts(f"ERROR: GENCODE bundle tree not found: {gencode_bundle}", file=sys.stderr)
        return 2

    parallel_dt_count = (
        min(max_parallel_data_types, len(data_types)) if parallel_data_types else 1
    )
    rows: list[dict[str, str]] = []
    all_ok = True
    sweep_t0 = time.perf_counter()
    log_ts(
        f"[sweep] start outdir={outdir} window_sizes={list(window_sizes)} "
        f"data_types={list(data_types)} parallel_data_types={parallel_data_types} "
        f"max_parallel_data_types={parallel_dt_count} max_parallel_tests={max_parallel_tests}"
    )

    script = Path(__file__).resolve()
    for ws in window_sizes:
        ws_t0 = time.perf_counter()
        log_ts(f"[sweep] window_size={ws} begin")
        materialize_stage1_config(
            base_config=base_stage1_config,
            window_size=ws,
            out_path=reports / "stage1_configs" / f"stage1.window_size_{ws}.yaml",
            n_workers=compute_stage1_n_workers_cap(
                parallel_data_types=parallel_dt_count,
                max_parallel_tests=max_parallel_tests,
            ),
        )

        if parallel_data_types and len(data_types) > 1:
            procs: list[tuple[str, subprocess.Popen[Any], Path]] = []
            for dt in data_types:
                log_path = logs / f"ws{ws}__{dt}.log"
                cmd = [
                    sys.executable,
                    str(script),
                    "--worker",
                    "--outdir",
                    str(outdir),
                    "--bundles-root",
                    str(bundles_root),
                    "--window-size",
                    str(ws),
                    "--data-type",
                    dt,
                    "--max-parallel-tests",
                    str(max_parallel_tests),
                    "--parallel-data-types-count",
                    str(parallel_dt_count),
                ]
                if base_stage1_config != default_stage1_config_path():
                    cmd.extend(["--stage1-config", str(base_stage1_config)])
                log_ts(f"[sweep] spawn worker {dt} ws={ws} log={log_path}")
                env = os.environ.copy()
                env.pop("TELOS_STAGE1_CACHE_DIR", None)
                with log_path.open("w", encoding="utf-8") as logfh:
                    logfh.write(f"# started_at={utc_now_iso()}\n# cmd={' '.join(cmd)}\n\n")
                    logfh.flush()
                    proc = subprocess.Popen(
                        cmd,
                        stdout=logfh,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=str(_repo_root()),
                    )
                procs.append((dt, proc, log_path))

            for dt, proc, log_path in procs:
                code = proc.wait()
                result_file = reports / f"worker_result__{run_id_for(data_type=dt, window_size=ws)}.json"
                if result_file.is_file():
                    row = json.loads(result_file.read_text(encoding="utf-8"))
                else:
                    row = {
                        "run_id": run_id_for(data_type=dt, window_size=ws),
                        "data_type": dt,
                        "window_size": str(ws),
                        "exit_code": str(code if code is not None else 1),
                        "status": "failed",
                        "error": f"worker exited without result file (log={log_path})",
                        "outdir": str(outdir / run_id_for(data_type=dt, window_size=ws)),
                        "summary_csv": "",
                        "benchmark_yaml": "",
                        "stage1_config": "",
                    }
                ok = str(row.get("exit_code")) == "0"
                all_ok = all_ok and ok
                rows.append(row)
                log_ts(
                    f"[sweep] worker done {row.get('run_id')} exit_code={row.get('exit_code')} "
                    f"log={log_path}"
                )
        else:
            for dt in data_types:
                row = run_benchmark_cell(
                    outdir=outdir,
                    bundles_root=bundles_root,
                    base_stage1_config=base_stage1_config,
                    data_type=dt,
                    window_size=ws,
                    max_parallel_tests=max_parallel_tests,
                    parallel_data_types=1,
                )
                all_ok = all_ok and str(row.get("exit_code")) == "0"
                rows.append(row)

        log_ts(f"[sweep] window_size={ws} end elapsed_s={elapsed_s(ws_t0)}")

    index_csv = reports / "feature_window_sweep_runs.csv"
    fieldnames = [
        "run_id",
        "data_type",
        "window_size",
        "exit_code",
        "status",
        "error",
        "elapsed_s",
        "started_at",
        "outdir",
        "summary_csv",
        "benchmark_yaml",
        "stage1_config",
        "stage1_n_workers",
        "max_parallel_tests",
    ]
    with index_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    log_ts(f"[sweep] complete elapsed_s={elapsed_s(sweep_t0)} index={index_csv}")
    return 0 if all_ok else 1


def main() -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description=(
            "Sweep Stage I feature window_size on human GENCODE bundles "
            "(train and test on GRCh38_gencode49)."
        )
    )
    p.add_argument(
        "--worker",
        action="store_true",
        help="Internal: run one (window_size, data_type) cell (used by subprocess workers).",
    )
    p.add_argument("--window-size", type=int, default=None, help="Worker: feature window size.")
    p.add_argument("--data-type", type=str, default=None, help="Worker: sr|cdna|drna|pacbio.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/human_gencode_feature_window"),
        help="Parent directory for per-run benchmark outputs.",
    )
    p.add_argument(
        "--bundles-root",
        type=Path,
        default=repo / "data" / "bundles",
        help="Root containing GRCh38_gencode49/ (default: <repo>/data/bundles).",
    )
    p.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        help="Base Stage I YAML to copy (default: stage1.defaults.yaml).",
    )
    p.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOW_SIZES),
        help=f"Feature window sizes to sweep (default: {list(DEFAULT_WINDOW_SIZES)}).",
    )
    p.add_argument(
        "--data-types",
        nargs="+",
        default=["sr", "cdna", "drna", "pacbio"],
        help="Subset of sr|cdna|drna|pacbio (default: all four).",
    )
    p.add_argument(
        "--max-parallel-tests",
        type=int,
        default=3,
        help="Concurrent benchmark test rows per cell (default: 3).",
    )
    p.add_argument(
        "--parallel-data-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run data types in parallel via subprocess (default: true).",
    )
    p.add_argument(
        "--max-parallel-data-types",
        type=int,
        default=4,
        help="Max concurrent data-type subprocesses per window size (default: 4).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-worker logs (default: <outdir>/reports/worker_logs).",
    )
    p.add_argument(
        "--parallel-data-types-count",
        type=int,
        default=None,
        help="Worker only: parent parallel data-type count for n_workers cap.",
    )
    args = p.parse_args()

    if args.worker:
        return _worker_main(args)

    base = args.stage1_config if args.stage1_config is not None else default_stage1_config_path()
    if not base.is_file():
        log_ts(f"ERROR: stage1 config not found: {base}", file=sys.stderr)
        return 2

    if args.max_parallel_tests < 1:
        log_ts("ERROR: --max-parallel-tests must be >= 1", file=sys.stderr)
        return 2
    if args.max_parallel_data_types < 1:
        log_ts("ERROR: --max-parallel-data-types must be >= 1", file=sys.stderr)
        return 2

    try:
        bundles_root = resolve_bundles_root(args.bundles_root)
    except FileNotFoundError as exc:
        log_ts(f"ERROR: {exc}", file=sys.stderr)
        return 2

    window_sizes = tuple(int(x) for x in args.window_sizes)
    if any(ws <= 0 for ws in window_sizes):
        log_ts("ERROR: window sizes must be positive integers", file=sys.stderr)
        return 2

    dtypes = tuple(str(x).strip().lower() for x in args.data_types)
    return run_sweep(
        outdir=args.outdir,
        bundles_root=bundles_root,
        base_stage1_config=base,
        data_types=dtypes,
        window_sizes=window_sizes,
        max_parallel_tests=args.max_parallel_tests,
        parallel_data_types=args.parallel_data_types,
        max_parallel_data_types=args.max_parallel_data_types,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
