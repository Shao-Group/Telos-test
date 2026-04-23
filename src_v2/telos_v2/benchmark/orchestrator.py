"""
Benchmark driver: one training phase (or reuse ``model_dir``), many prediction rows, one summary CSV.

This module is the programmatic core behind ``telos v2 benchmark`` and ``benchmark-matrix``; it does
not parse CLI flags—callers pass :class:`~telos_v2.config_models.BenchmarkIO`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telos_v2.benchmark.report import stub_test_row, write_benchmark_summary_csv
from telos_v2.benchmark.stage1_metrics import evaluate_stage1_test_aupr
from telos_v2.benchmark.transcript_pr_runner import (
    merge_transcript_pr_into_row,
    run_backend_transcript_pr,
)
from telos_v2.benchmark.util import as_path, round_float_metrics_in_row
from telos_v2.commands.predict import run_predict
from telos_v2.commands.train import run_train
from telos_v2.config_loader import default_stage1_config_path, get_nested, load_mapping_config
from telos_v2.config_validation import validate_benchmark_config, validate_stage_config
from telos_v2.config_models import BenchmarkIO, PredictIO, TrainIO
from telos_v2.evaluation.benchmark_sklearn_metrics import (
    sklearn_metrics_ranked_vs_bundle_tmap,
    suffix_keys,
)
from telos_v2.models import TRANSCRIPTS_RANKED_RF_TSV, TRANSCRIPTS_RANKED_XGB_TSV
from telos_v2.models.chrom_split import parse_split_policy
from telos_v2.validation.preflight import (
    PreflightError,
    ensure_run_layout,
    run_preflight_benchmark,
)


def run_benchmark(cfg: BenchmarkIO) -> int:
    """
    Execute a full benchmark described by a YAML/JSON mapping on disk.

    **Phase 0 — validation**

    - :func:`~telos_v2.validation.preflight.run_preflight_benchmark` on ``cfg.config``.
    - :func:`~telos_v2.validation.preflight.ensure_run_layout` for ``cfg.outdir`` (reports dir).
    - Load mapping; :func:`~telos_v2.config_validation.validate_benchmark_config`.

    **Phase 1 — training**

    - Read ``train.mode``: ``run`` builds :class:`~telos_v2.config_models.TrainIO` and calls
      :func:`~telos_v2.commands.train.run_train`; ``skip`` requires ``train.model_dir`` and never trains.
    - Default train output directory is ``cfg.outdir / "train"`` if YAML omits ``train.outdir``.
    - On training failure (non-zero code), aborts before any test with process exit ``1``.

    **Phase 2 — tests loop**

    For each ``tests[]`` entry: build :class:`~telos_v2.config_models.PredictIO` (per-test ``outdir``,
    ``config``, ``model_dir`` override train defaults), run
    :func:`~telos_v2.commands.predict.run_predict`, and record paths to ranked TSVs and status.

    **Phase 3 — analysis (optional)**

    Controlled by ``analysis.enabled`` and nested blocks:

    - **Stage I test AUPR** when predict succeeds and ``tmap`` + ``ref_gtf`` exist: joins scored sites
      to coverage and reference sites (:func:`~telos_v2.benchmark.stage1_metrics.evaluate_stage1_test_aupr`).
    - **Sklearn on static tmap** only when ``analysis.enabled`` and transcript PR is disabled: fast
      metrics from ranked TSV × bundle tmap.
    - **Transcript PR (gffcompare path)** when ``pr_vs_baseline.enabled``: for each backend, runs
      :func:`~telos_v2.benchmark.transcript_pr_runner.run_backend_transcript_pr` and merges columns.
      ``benchmark_mode`` + ``debug.keep_pr_work`` control whether PR intermediates are ephemeral.

    **Phase 4 — summary**

    - :func:`~telos_v2.benchmark.report.write_benchmark_summary_csv` to ``reports/benchmark_summary.csv``.
    - Exit ``0`` only if every test row has ``status == "ok"``; otherwise ``1``. Config/preflight errors use ``2`` earlier in the train/predict commands; this function returns ``1`` on partial test failure.

    Args:
        cfg: Benchmark config path and output root.

    Returns:
        Shell-friendly integer exit status.
    """
    try:
        run_preflight_benchmark(cfg.config)
    except PreflightError as exc:
        print(f"[telos_v2] preflight failed: {exc}")
        return 2

    layout = ensure_run_layout(cfg.outdir)
    try:
        bench_cfg = load_mapping_config(cfg.config)
        validate_benchmark_config(bench_cfg)
    except ValueError as exc:
        print(f"[telos_v2] benchmark config error: {exc}")
        return 2

    train_obj = get_nested(bench_cfg, ["train"], None)
    tests = get_nested(bench_cfg, ["tests"], None)
    if not isinstance(train_obj, dict):
        print("[telos_v2] benchmark config must contain a 'train' mapping.")
        return 2
    if not isinstance(tests, list) or not tests:
        print("[telos_v2] benchmark config must contain non-empty 'tests' list.")
        return 2

    analysis_block = get_nested(bench_cfg, ["analysis"], {}) or {}
    bench_mode = str(analysis_block.get("benchmark_mode", "minimal")).strip().lower()
    if bench_mode not in ("minimal", "full"):
        print("[telos_v2] benchmark config error: analysis.benchmark_mode must be 'minimal' or 'full'.")
        return 2
    debug_block = analysis_block.get("debug") if isinstance(analysis_block.get("debug"), dict) else {}
    keep_pr_work = bool(debug_block.get("keep_pr_work", False))

    root = layout.root.resolve()
    print(f"[telos_v2] benchmark output root: {root}")
    print(f"[telos_v2]   summaries -> {layout.reports_dir.resolve()}/")
    print(
        f"[telos_v2]   analysis mode: {bench_mode}"
        + (" (PR work kept under reports/pr/)" if bench_mode == "full" or keep_pr_work else " (ephemeral PR workdir)")
    )
    print(
        f"[telos_v2]   per-test outputs -> {root}/tests/<id>/ "
        f"(train may use train.outdir from YAML under this tree or elsewhere)"
    )

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
            print(f"[telos_v2] benchmark config error: {exc}")
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
        print(f"[telos_v2]   train skipped; using models -> {model_dir.resolve()}/")
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
            print(f"[telos_v2] benchmark config error: {exc}")
            return 2
        print("[telos_v2] benchmark: train baseline model")
        train_code = run_train(train_io)
        model_dir = train_out / "models"
        err_hint = ""
        if train_code != 0:
            err_hint = (
                "Training failed; see terminal lines prefixed with [telos_v2] above. "
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
            print(f"[telos_v2] benchmark aborted: training failed (code={train_code})")
            return 1
        print(f"[telos_v2]   training run -> {train_out.resolve()}/")

    for tr in train_rows:
        round_float_metrics_in_row(tr)

    rows: list[dict[str, Any]] = []
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
            print(
                f"[telos_v2] benchmark: chromosomes_file not found ({cp}); "
                "using automatic validation-chrom filter from stage1.training.split_policy."
            )

    pr_plot_effective = bool(bench_mode == "full" and pr_plot_cfg)
    pr_save_tables_effective = bool(bench_mode == "full" and pr_save_tables_cfg)
    ephemeral_pr = bool(bench_mode == "minimal" and not keep_pr_work)

    sklearn_bundle_tmap = analyze and not pr_vs_baseline

    def _stage2_aupr_from_ranked_vs_tmap_and_cov(
        ranked_tsv: Path, tmap_path: Path, assembly_gtf: Path
    ) -> dict[str, float | int | None]:
        """
        Stage II evaluation used by the original training flow: AUPR(model score vs tmap label)
        and AUPR(baseline coverage vs tmap label) on the same transcript_id set.

        This differs from transcript PR (gffcompare re-run + gtfcuff auc). It is intended to be
        a fast, label-aligned sanity check where a reasonable model should beat abundance baseline.
        """
        try:
            import pandas as pd
            from sklearn.metrics import average_precision_score
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(f"Stage II AUPR requires pandas+sklearn: {exc}") from exc

        from telos_v2.backends.gtfformat import build_cov_dataframe
        from telos_v2.labels.transcript_labels import load_tmap_labels_with_ref

        ranked = pd.read_csv(ranked_tsv, sep="\t", dtype={"transcript_id": str})
        if "transcript_id" not in ranked.columns or "pred_prob" not in ranked.columns:
            raise ValueError(f"ranked TSV missing transcript_id/pred_prob: {ranked_tsv}")
        ranked_rows_raw = int(len(ranked))
        ranked_dup_tx = int(ranked.duplicated("transcript_id").sum())
        ranked = ranked[["transcript_id", "pred_prob"]].copy()
        ranked["pred_prob"] = pd.to_numeric(ranked["pred_prob"], errors="coerce")
        ranked = ranked.dropna(subset=["pred_prob"]).drop_duplicates("transcript_id", keep="first")

        labels = load_tmap_labels_with_ref(tmap_path)
        labels_rows_raw = int(len(labels))
        labels_dup_tx = int(labels.duplicated("transcript_id").sum())
        labels = labels.drop_duplicates("transcript_id", keep="first")
        cov = build_cov_dataframe(assembly_gtf)
        if cov.empty:
            return None, None
        cov = cov[["transcript_id", "coverage"]].copy()
        cov_rows_raw = int(len(cov))
        cov_dup_tx = int(cov.duplicated("transcript_id").sum())
        cov["coverage"] = pd.to_numeric(cov["coverage"], errors="coerce")
        cov = cov.dropna(subset=["coverage"])
        # Deterministic transcript-level baseline score: keep max abundance when duplicates exist.
        cov = cov.groupby("transcript_id", as_index=False, sort=False)["coverage"].max()

        df = ranked.merge(labels, on="transcript_id", how="inner").merge(cov, on="transcript_id", how="inner")
        if df.empty or df["label"].nunique() < 2:
            return {
                "aupr_model": None,
                "aupr_baseline": None,
                "aupr_model_novel": None,
                "aupr_baseline_novel": None,
                "n_joined": int(len(df)),
                "n_pos_novel": int(
                    (
                        (df["label"].astype(int) == 1)
                        & df["ref_id"].astype(str).str.startswith("NOVEL_TX_")
                    ).sum()
                )
                if len(df) > 0
                else 0,
            }
        print(
            "[telos_v2] benchmark stage2 AUPR debug: "
            f"ranked_rows={ranked_rows_raw} ranked_dup_tx={ranked_dup_tx} "
            f"labels_rows={labels_rows_raw} labels_dup_tx={labels_dup_tx} "
            f"cov_rows={cov_rows_raw} cov_dup_tx={cov_dup_tx} "
            f"joined_rows={len(df)} pos_labels={int(df['label'].sum())}"
        )

        y = df["label"].astype(int)
        aupr_model = float(average_precision_score(y, df["pred_prob"].astype(float)))
        aupr_base = float(average_precision_score(y, df["coverage"].astype(float)))

        y_novel = (
            (df["label"].astype(int) == 1)
            & df["ref_id"].astype(str).str.startswith("NOVEL_TX_")
        ).astype(int)
        if y_novel.nunique() >= 2:
            aupr_model_novel = float(average_precision_score(y_novel, df["pred_prob"].astype(float)))
            aupr_base_novel = float(average_precision_score(y_novel, df["coverage"].astype(float)))
        else:
            aupr_model_novel = None
            aupr_base_novel = None
        return {
            "aupr_model": aupr_model,
            "aupr_baseline": aupr_base,
            "aupr_model_novel": aupr_model_novel,
            "aupr_baseline_novel": aupr_base_novel,
            "n_joined": int(len(df)),
            "n_pos_novel": int(y_novel.sum()),
        }

    for i, test_obj in enumerate(tests, start=1):
        if not isinstance(test_obj, dict):
            rows.append(
                stub_test_row(
                    f"test_{i}",
                    status="invalid_config",
                    error="test entry is not a mapping",
                )
            )
            if stop_on_error:
                break
            continue

        test_id = str(test_obj.get("id", f"test_{i}"))
        try:
            predict_io = PredictIO(
                bam=as_path(test_obj.get("bam"), f"tests[{i}].bam"),
                gtf=as_path(test_obj.get("gtf"), f"tests[{i}].gtf"),
                model_dir=(
                    Path(str(test_obj["model_dir"]))
                    if test_obj.get("model_dir")
                    else model_dir
                ),
                outdir=(
                    Path(str(test_obj["outdir"]))
                    if test_obj.get("outdir")
                    else (cfg.outdir / "tests" / test_id)
                ),
                config_file=(
                    Path(str(test_obj["config"]))
                    if test_obj.get("config")
                    else train_config
                ),
            )
        except ValueError as exc:
            rows.append(
                stub_test_row(
                    test_id,
                    status="invalid_config",
                    error=str(exc),
                )
            )
            if stop_on_error:
                break
            continue

        print(f"[telos_v2] benchmark {i}/{len(tests)}: predict {test_id}")
        predict_code = run_predict(predict_io)
        ranked_rf = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_RF_TSV
        ranked_xgb = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_XGB_TSV
        sites_scored = predict_io.outdir / "predictions" / "sites.scored.tsv"
        aid = test_obj.get("assembler_id")
        row: dict[str, Any] = {
            "test_id": test_id,
            "assembler_id": str(aid).strip() if aid is not None and str(aid).strip() else "",
            "predict_code": str(predict_code),
            "status": "ok" if predict_code == 0 else "predict_failed",
            "error": "",
            "predict_outdir": str(predict_io.outdir),
            "ranked_rf_tsv": str(ranked_rf) if ranked_rf.exists() else "",
            "ranked_xgb_tsv": str(ranked_xgb) if ranked_xgb.exists() else "",
        }
        if predict_code == 0 and test_obj.get("tmap"):
            ref_gtf_raw = test_obj.get("ref_gtf")
            # Stage II AUPR sanity check (ranked TSV vs static bundle tmap labels, plus coverage baseline).
            # This is the metric legacy code typically expects the model to beat.
            try:
                tmap_path = Path(str(test_obj["tmap"]))
                assembly_gtf = Path(str(test_obj["gtf"]))
                for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                    m = _stage2_aupr_from_ranked_vs_tmap_and_cov(rp, tmap_path, assembly_gtf)
                    row[f"stage2_test_aupr_{suf}"] = m["aupr_model"]
                    row[f"stage2_test_aupr_novel_{suf}"] = m["aupr_model_novel"]
                    row["stage2_test_n_eval_tx"] = m["n_joined"]
                    row["stage2_test_n_novel_pos_tx"] = m["n_pos_novel"]
                    if "stage2_test_aupr_baseline" not in row:
                        row["stage2_test_aupr_baseline"] = m["aupr_baseline"]
                    if "stage2_test_aupr_novel_baseline" not in row:
                        row["stage2_test_aupr_novel_baseline"] = m["aupr_baseline_novel"]
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                row["analysis_error"] = str(exc)

            if ref_gtf_raw and sites_scored.is_file():
                try:
                    pred_cfg = load_mapping_config(predict_io.config_file)
                    validate_stage_config(pred_cfg)
                except ValueError as exc:
                    row["error"] = f"stage config invalid: {exc}"
                    pred_cfg = {}
                tol = int(get_nested(pred_cfg, ["stage1", "training", "site_label_tolerance_bp"], 50))
                try:
                    st1 = evaluate_stage1_test_aupr(
                        sites_scored,
                        Path(str(test_obj["gtf"])),
                        Path(str(ref_gtf_raw)),
                        tolerance_bp=tol,
                    )
                    row.update(st1)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    row["error"] = f"stage1_test_aupr_failed: {exc}"

            if sklearn_bundle_tmap:
                try:
                    tmap_path = Path(str(test_obj["tmap"]))
                    for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                        m = sklearn_metrics_ranked_vs_bundle_tmap(rp, tmap_path)
                        row.update(suffix_keys(m, suf))
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    row["analysis_error"] = str(exc)

            if pr_vs_baseline:
                try:
                    pred_cfg = load_mapping_config(predict_io.config_file)
                    validate_stage_config(pred_cfg)
                except ValueError as exc:
                    row["transcript_pr_error"] = f"stage config invalid: {exc}"
                    pred_cfg = {}
                pr_reports = predict_io.outdir / "reports" / "pr"
                measure = str(pr_block.get("measure", "cov"))
                gffcompare_ex = pr_block.get("gffcompare_bin")
                gffcompare_bin = str(gffcompare_ex).strip() if gffcompare_ex else None
                split_pol = str(
                    get_nested(pred_cfg, ["stage1", "training", "split_policy"], "chr1-10")
                )
                pr_errs: list[str] = []
                ref_gtf_raw = test_obj.get("ref_gtf")
                if not ref_gtf_raw:
                    pr_errs.append(
                        "benchmark transcript PR requires test.ref_gtf (reference annotation for gffcompare). "
                        "Regenerate YAML with benchmark-matrix or add ref_gtf to each test."
                    )
                try:
                    autosome_train_range = parse_split_policy(split_pol)
                except ValueError:
                    autosome_train_range = None
                    pr_errs.append(
                        f"Invalid stage1.training.split_policy for PR chrom filter: {split_pol!r}"
                    )
                if pr_errs:
                    row["transcript_pr_error"] = "; ".join(pr_errs)
                else:
                    if not row.get("transcript_pr_error"):
                        assembly_gtf = Path(str(test_obj["gtf"]))
                        ref_gtf = Path(str(ref_gtf_raw))
                        try:
                            row["transcript_pr_reports_dir"] = str(pr_reports.resolve())
                            for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                                pr_row = run_backend_transcript_pr(
                                    assembly_gtf=assembly_gtf,
                                    ref_gtf=ref_gtf,
                                    ranked_tsv=rp,
                                    reports_pr_dir=pr_reports,
                                    test_id=test_id,
                                    backend_suffix=suf,
                                    gffcompare_bin=gffcompare_bin,
                                    measure=measure,
                                    plot=pr_plot_effective,
                                    save_pr_tables=pr_save_tables_effective,
                                    chromosomes_path=pr_chromosomes_path,
                                    filter_validation_chroms=pr_filter_val_chroms,
                                    autosome_train_range=autosome_train_range,
                                    ephemeral_workdir=ephemeral_pr,
                                )
                                merge_transcript_pr_into_row(row, pr_row, suf)
                                if "transcript_pr_auc_baseline" not in row:
                                    row["transcript_pr_auc_baseline"] = pr_row.get(
                                        "transcript_pr_auc_baseline"
                                    )
                            pr_bits = []
                            for suf, lab in (("rf", "RF"), ("xgb", "XGB")):
                                v = row.get(f"transcript_pr_auc_model_{suf}")
                                if isinstance(v, (int, float)):
                                    pr_bits.append(f"{lab} transcript_pr_auc={float(v):.2f}")
                            print(
                                f"[telos_v2] benchmark transcript PR {test_id}: "
                                + (", ".join(pr_bits) if pr_bits else "no AUC")
                                + (f" dir={pr_reports}" if not ephemeral_pr else " (ephemeral workdir)")
                            )
                        except (
                            FileNotFoundError,
                            OSError,
                            ValueError,
                            KeyError,
                            RuntimeError,
                        ) as exc:
                            row["transcript_pr_error"] = str(exc)

        round_float_metrics_in_row(row)
        rows.append(row)
        if predict_code != 0 and stop_on_error:
            break

    summary_csv = layout.reports_dir / "benchmark_summary.csv"
    write_benchmark_summary_csv(rows, summary_csv)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print("[telos_v2] benchmark complete")
    print(f"  config={cfg.config}")
    print(f"  benchmark_summary_csv={summary_csv.resolve()}")
    print(f"  completed_ok={ok}/{len(rows)}")
    return 0 if ok == len(rows) else 1
