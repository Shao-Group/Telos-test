"""
Benchmark driver: one training phase (or reuse ``model_dir``), many prediction rows, one summary CSV.

This module is the programmatic core behind ``telos v2 benchmark`` and ``benchmark-matrix``; it does
not parse CLI flags—callers pass :class:`~telos.config_models.BenchmarkIO`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from telos_repro.benchmark.orchestrator_tests import BenchmarkTestJob, execute_benchmark_test
from telos_repro.benchmark.report import write_benchmark_summary_csv
from telos_repro.benchmark.timing import append_timing_row, elapsed_s, log_ts, timed_phase, utc_now_iso
from telos_repro.benchmark.util import as_path, round_float_metrics_in_row
from telos_repro.backend import predict as run_predict
from telos_repro.backend import train as run_train
from telos_repro.config_loader import default_stage1_config_path, get_nested, load_mapping_config
from telos.config_validation import validate_benchmark_config, validate_stage_config
from telos.config_models import BenchmarkIO, TrainIO
from telos.validation.preflight import (
    PreflightError,
    ensure_run_layout,
    run_preflight_benchmark,
)


def run_benchmark(cfg: BenchmarkIO) -> int:
    """
    Execute a full benchmark described by a YAML/JSON mapping on disk.

    **Phase 0 — validation**

    - :func:`~telos.validation.preflight.run_preflight_benchmark` on ``cfg.config``.
    - :func:`~telos.validation.preflight.ensure_run_layout` for ``cfg.outdir`` (reports dir).
    - Load mapping; :func:`~telos.config_validation.validate_benchmark_config`.

    **Phase 1 — training**

    - Read ``train.mode``: ``run`` builds :class:`~telos.config_models.TrainIO` and calls
      :func:`~telos_repro.commands.train.run_train`; ``skip`` requires ``train.model_dir`` and never trains.
    - Default train output directory is ``cfg.outdir / "train"`` if YAML omits ``train.outdir``.
    - On training failure (non-zero code), aborts before any test with process exit ``1``.

    **Phase 2 — tests loop**

    For each ``tests[]`` entry: build :class:`~telos.config_models.PredictIO` (per-test ``outdir``,
    ``config``, ``model_dir`` override train defaults), run
    :func:`~telos_repro.commands.predict.run_predict`, and record paths to ranked TSVs and status.

    **Phase 3 — analysis (optional)**

    Controlled by ``analysis.enabled`` and nested blocks:

    - **Stage I test AUPR** when predict succeeds and ``tmap`` + ``ref_gtf`` exist: joins scored sites
      to coverage and reference sites (:func:`~telos_repro.benchmark.stage1_metrics.evaluate_stage1_test_aupr`).
    - **Sklearn on static tmap** only when ``analysis.enabled`` and transcript PR is disabled: fast
      metrics from ranked TSV × bundle tmap.
    - **Transcript PR (gffcompare path)** when ``pr_vs_baseline.enabled``: for each backend, runs
      :func:`~telos_repro.benchmark.transcript_pr_runner.run_backend_transcript_pr` and merges columns.
      ``benchmark_mode`` + ``debug.keep_pr_work`` control whether PR intermediates are ephemeral.

    **Phase 4 — summary**

    - :func:`~telos_repro.benchmark.report.write_benchmark_summary_csv` to ``reports/benchmark_summary.csv``.
    - Exit ``0`` only if every test row has ``status == "ok"``; otherwise ``1``. Config/preflight errors use ``2`` earlier in the train/predict commands; this function returns ``1`` on partial test failure.

    Args:
        cfg: Benchmark config path and output root.

    Returns:
        Shell-friendly integer exit status.
    """
    timing_csv = (cfg.outdir / "reports" / "benchmark_timing.csv").resolve()

    try:
        run_preflight_benchmark(cfg.config)
    except PreflightError as exc:
        log_ts(f"[telos_repro] preflight failed: {exc}")
        return 2

    layout = ensure_run_layout(cfg.outdir)
    try:
        bench_cfg = load_mapping_config(cfg.config)
        validate_benchmark_config(bench_cfg)
    except ValueError as exc:
        log_ts(f"[telos_repro] benchmark config error: {exc}")
        return 2

    train_obj = get_nested(bench_cfg, ["train"], None)
    tests = get_nested(bench_cfg, ["tests"], None)
    if not isinstance(train_obj, dict):
        log_ts("[telos_repro] benchmark config must contain a 'train' mapping.")
        return 2
    if not isinstance(tests, list) or not tests:
        log_ts("[telos_repro] benchmark config must contain non-empty 'tests' list.")
        return 2

    analysis_block = get_nested(bench_cfg, ["analysis"], {}) or {}
    bench_mode = str(analysis_block.get("benchmark_mode", "minimal")).strip().lower()
    if bench_mode not in ("minimal", "full"):
        log_ts("[telos_repro] benchmark config error: analysis.benchmark_mode must be 'minimal' or 'full'.")
        return 2
    debug_block = analysis_block.get("debug") if isinstance(analysis_block.get("debug"), dict) else {}
    keep_pr_work = bool(debug_block.get("keep_pr_work", False))

    root = layout.root.resolve()
    max_parallel_tests = int(get_nested(bench_cfg, ["execution", "max_parallel_tests"], 1))
    if max_parallel_tests < 1:
        max_parallel_tests = 1

    log_ts(f"[telos_repro] benchmark output root: {root}")
    log_ts(f"[telos_repro]   summaries -> {layout.reports_dir.resolve()}/")
    log_ts(f"[telos_repro]   timing log -> {timing_csv}")
    log_ts(
        f"[telos_repro]   analysis mode: {bench_mode}"
        + (" (PR work kept under reports/pr/)" if bench_mode == "full" or keep_pr_work else " (ephemeral PR workdir)")
    )
    log_ts(
        f"[telos_repro]   per-test outputs -> {root}/tests/<id>/ "
        f"(train may use train.outdir from YAML under this tree or elsewhere)"
    )
    log_ts(f"[telos_repro]   max_parallel_tests={max_parallel_tests}")

    stop_on_error = bool(get_nested(bench_cfg, ["execution", "stop_on_error"], False))
    train_rows: list[dict[str, Any]] = []
    train_mode = str(get_nested(train_obj, ["mode"], "run")).strip().lower()
    train_config = (
        Path(str(train_obj["config"]))
        if train_obj.get("config")
        else default_stage1_config_path()
    )
    train_out = (
        Path(str(train_obj["outdir"]))
        if train_obj.get("outdir")
        else (cfg.outdir / "train")
    )

    model_dir: Path
    if train_mode == "skip":
        try:
            model_dir = as_path(train_obj.get("model_dir"), "train.model_dir")
        except ValueError as exc:
            log_ts(f"[telos_repro] benchmark config error: {exc}")
            return 2
        train_rows.append(
            {
                "status": "skipped",
                "train_code": "",
                "train_outdir": str(train_out),
                "model_dir": str(model_dir),
                "error": "",
            }
        )
        log_ts(f"[telos_repro]   train skipped; using models -> {model_dir.resolve()}/")
    else:
        try:
            train_io = TrainIO(
                bam=as_path(train_obj.get("bam"), "train.bam"),
                gtf=as_path(train_obj.get("gtf"), "train.gtf"),
                ref_gtf=as_path(train_obj.get("ref_gtf"), "train.ref_gtf"),
                tmap=as_path(train_obj.get("tmap"), "train.tmap"),
                gtf_pool=tuple(Path(str(p)) for p in (train_obj.get("gtf_pool") or [])) or None,
                tmap_pool=tuple(Path(str(p)) for p in (train_obj.get("tmap_pool") or [])) or None,
                outdir=train_out,
                config_file=train_config,
            )
        except ValueError as exc:
            log_ts(f"[telos_repro] benchmark config error: {exc}")
            return 2
        log_ts("[telos_repro] benchmark: train baseline model")
        with timed_phase(phase="train", timing_csv=timing_csv):
            train_code = run_train(train_io)
        model_dir = train_out / "models"
        err_hint = ""
        if train_code != 0:
            err_hint = (
                "Training failed; see terminal lines prefixed with [telos_repro] above. "
                "Exit 2: preflight, candidates, cov, Stage I/II error, etc.; exit 3: no candidates."
            )
        train_rows.append(
            {
                "status": "ok" if train_code == 0 else "failed",
                "train_code": str(train_code),
                "train_outdir": str(train_out),
                "model_dir": str(model_dir),
                "error": err_hint,
            }
        )
        if train_code != 0:
            log_ts(f"[telos_repro] benchmark aborted: training failed (code={train_code})")
            return 1
        log_ts(f"[telos_repro]   training run -> {train_out.resolve()}/")

    for tr in train_rows:
        round_float_metrics_in_row(tr)

    analyze = bool(get_nested(bench_cfg, ["analysis", "enabled"], True))
    pr_vs_baseline = bool(get_nested(bench_cfg, ["analysis", "pr_vs_baseline", "enabled"], True))
    pr_plot_cfg = bool(get_nested(bench_cfg, ["analysis", "pr_vs_baseline", "plot"], False))
    pr_block = get_nested(bench_cfg, ["analysis", "pr_vs_baseline"], {}) or {}
    pr_save_tables_cfg = bool(pr_block.get("save_pr_tables", False))
    pr_filter_val_chroms = bool(pr_block.get("filter_validation_chroms", True))
    pr_chrom_raw = pr_block.get("chromosomes_file")
    pr_chromosomes_path: Path | None = None
    if pr_chrom_raw is not None and str(pr_chrom_raw).strip().lower() not in ("", "null", "none"):
        cp = Path(str(pr_chrom_raw).strip())
        if cp.is_file():
            pr_chromosomes_path = cp
        else:
            log_ts(
                f"[telos_repro] benchmark: chromosomes_file not found ({cp}); "
                "using automatic validation-chrom filter from stage1.training.split_policy."
            )

    pr_plot_effective = bool(bench_mode == "full" and pr_plot_cfg)
    # Keep compact PR point tables even in minimal mode so downstream plotting is fast/reproducible.
    pr_save_tables_effective = bool(pr_vs_baseline and (pr_save_tables_cfg or True))
    ephemeral_pr = bool(bench_mode == "minimal" and not keep_pr_work)

    sklearn_bundle_tmap = analyze and not pr_vs_baseline
    n_tests = len(tests)

    jobs: list[BenchmarkTestJob] = []
    for i, test_obj in enumerate(tests, start=1):
        jobs.append(
            BenchmarkTestJob(
                test_index=i,
                n_tests=n_tests,
                test_obj=test_obj if isinstance(test_obj, dict) else {},
                model_dir=model_dir,
                bench_outdir=cfg.outdir,
                train_config=train_config,
                analyze=analyze,
                sklearn_bundle_tmap=sklearn_bundle_tmap,
                pr_vs_baseline=pr_vs_baseline,
                pr_plot_effective=pr_plot_effective,
                pr_save_tables_effective=pr_save_tables_effective,
                pr_filter_val_chroms=pr_filter_val_chroms,
                pr_chromosomes_path=pr_chromosomes_path,
                ephemeral_pr=ephemeral_pr,
                pr_block=pr_block if isinstance(pr_block, dict) else {},
                timing_csv=timing_csv,
            )
        )

    rows: list[dict[str, Any]] = []
    tests_t0 = time.perf_counter()
    log_ts(f"[telos_repro] benchmark tests phase start n_tests={n_tests} max_parallel={max_parallel_tests}")

    if max_parallel_tests <= 1:
        for job in jobs:
            row = execute_benchmark_test(job)
            rows.append(row)
            if stop_on_error and row.get("status") != "ok":
                break
    else:
        workers = min(max_parallel_tests, n_tests)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute_benchmark_test, job): job for job in jobs}
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                if stop_on_error and row.get("status") != "ok":
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
        rows.sort(key=lambda r: int(r.get("_test_index", 0)))

    for row in rows:
        row.pop("_test_index", None)

    tests_elapsed = elapsed_s(tests_t0)
    log_ts(f"[telos_repro] benchmark tests phase end elapsed_s={tests_elapsed}")
    append_timing_row(
        timing_csv,
        {
            "phase": "tests_all",
            "elapsed_s": str(tests_elapsed),
            "n_tests": str(n_tests),
            "max_parallel_tests": str(max_parallel_tests),
            "ended_at": utc_now_iso(),
        },
    )

    summary_csv = layout.reports_dir / "benchmark_summary.csv"
    write_benchmark_summary_csv(rows, summary_csv)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    log_ts("[telos_repro] benchmark complete")
    log_ts(f"  config={cfg.config}")
    log_ts(f"  benchmark_summary_csv={summary_csv.resolve()}")
    log_ts(f"  benchmark_timing_csv={timing_csv}")
    log_ts(f"  completed_ok={ok}/{len(rows)}")
    return 0 if ok == len(rows) else 1
