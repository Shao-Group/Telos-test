"""
Single benchmark test execution (predict + analysis) for sequential or parallel orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from telos_repro.benchmark.stage1_metrics import evaluate_stage1_test_aupr
from telos_repro.benchmark.timing import elapsed_s, log_ts, utc_now_iso
from telos_repro.benchmark.transcript_pr_runner import (
    merge_transcript_pr_into_row,
    run_backend_transcript_pr,
)
from telos_repro.benchmark.util import as_path, round_float_metrics_in_row
from telos_repro.backend import predict as run_predict
from telos_repro.config_loader import get_nested, load_mapping_config
from telos.config_models import PredictIO
from telos.config_validation import validate_stage_config
from telos_repro.evaluation.benchmark_sklearn_metrics import (
    sklearn_metrics_ranked_vs_bundle_tmap,
    suffix_keys,
)
from telos.models import TRANSCRIPTS_RANKED_RF_TSV, TRANSCRIPTS_RANKED_XGB_TSV
from telos.models.chrom_split import parse_split_policy


def stage2_aupr_from_ranked_vs_tmap_and_cov(
    ranked_tsv: Path, tmap_path: Path, assembly_gtf: Path
) -> dict[str, float | int | None]:
    try:
        from sklearn.metrics import average_precision_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Stage II AUPR requires pandas+sklearn: {exc}") from exc

    from telos.backends.gtfformat import build_cov_dataframe
    from telos.labels.transcript_labels import load_tmap_labels_with_ref

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
        return {
            "aupr_model": None,
            "aupr_baseline": None,
            "aupr_model_novel": None,
            "aupr_baseline_novel": None,
            "n_joined": 0,
            "n_pos_novel": 0,
        }
    cov = cov[["transcript_id", "coverage"]].copy()
    cov_rows_raw = int(len(cov))
    cov_dup_tx = int(cov.duplicated("transcript_id").sum())
    cov["coverage"] = pd.to_numeric(cov["coverage"], errors="coerce")
    cov = cov.dropna(subset=["coverage"])
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
    log_ts(
        "[telos_repro] benchmark stage2 AUPR debug: "
        f"ranked_rows={ranked_rows_raw} ranked_dup_tx={ranked_dup_tx} "
        f"labels_rows={labels_rows_raw} labels_dup_tx={labels_dup_tx} "
        f"cov_rows={cov_rows_raw} cov_dup_tx={cov_dup_tx} "
        f"joined_rows={len(df)} pos_labels={int(df['label'].sum())}"
    )

    y = df["label"].astype(int)
    aupr_model = float(average_precision_score(y, df["pred_prob"].astype(float)))
    aupr_base = float(average_precision_score(y, df["coverage"].astype(float)))

    y_novel = (
        (df["label"].astype(int) == 1) & df["ref_id"].astype(str).str.startswith("NOVEL_TX_")
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


def write_stage2_curve_points(
    *,
    ranked_rf_tsv: Path,
    ranked_xgb_tsv: Path,
    tmap_path: Path,
    assembly_gtf: Path,
    out_path: Path,
) -> None:
    from telos.backends.gtfformat import build_cov_dataframe
    from telos.labels.transcript_labels import load_tmap_labels_with_ref

    rf = pd.read_csv(ranked_rf_tsv, sep="\t", dtype={"transcript_id": str})
    xgb = pd.read_csv(ranked_xgb_tsv, sep="\t", dtype={"transcript_id": str})
    if "pred_prob" not in rf.columns or "pred_prob" not in xgb.columns:
        return
    rf = (
        rf[["transcript_id", "pred_prob"]]
        .rename(columns={"pred_prob": "score_rf"})
        .dropna(subset=["score_rf"])
        .drop_duplicates("transcript_id", keep="first")
    )
    xgb = (
        xgb[["transcript_id", "pred_prob"]]
        .rename(columns={"pred_prob": "score_xgb"})
        .dropna(subset=["score_xgb"])
        .drop_duplicates("transcript_id", keep="first")
    )
    labels = load_tmap_labels_with_ref(tmap_path).drop_duplicates("transcript_id", keep="first")
    cov = build_cov_dataframe(assembly_gtf)
    if cov.empty:
        return
    cov = cov[["transcript_id", "coverage"]].copy()
    cov["coverage"] = pd.to_numeric(cov["coverage"], errors="coerce")
    cov = cov.dropna(subset=["coverage"]).groupby("transcript_id", as_index=False)["coverage"].max()
    merged = rf.merge(xgb, on="transcript_id", how="inner").merge(labels, on="transcript_id", how="inner").merge(
        cov, on="transcript_id", how="inner"
    )
    if merged.empty:
        return
    out = pd.DataFrame(
        {
            "label": merged["label"].astype(int).values,
            "is_novel": merged["ref_id"].astype(str).str.startswith("NOVEL_TX_").astype(int).values,
            "score_rf": pd.to_numeric(merged["score_rf"], errors="coerce").values,
            "score_xgb": pd.to_numeric(merged["score_xgb"], errors="coerce").values,
            "score_baseline_cov": pd.to_numeric(merged["coverage"], errors="coerce").values,
        }
    ).dropna(subset=["score_rf", "score_xgb", "score_baseline_cov"])
    if out.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)


@dataclass(frozen=True)
class BenchmarkTestJob:
    test_index: int
    n_tests: int
    test_obj: dict[str, Any]
    model_dir: Path
    bench_outdir: Path
    train_config: Path
    analyze: bool
    sklearn_bundle_tmap: bool
    pr_vs_baseline: bool
    pr_plot_effective: bool
    pr_save_tables_effective: bool
    pr_filter_val_chroms: bool
    pr_chromosomes_path: Path | None
    ephemeral_pr: bool
    pr_block: dict[str, Any]
    timing_csv: Path | None


def execute_benchmark_test(job: BenchmarkTestJob) -> dict[str, Any]:
    """Run predict + optional analysis for one benchmark test row."""
    i = job.test_index
    test_obj = job.test_obj
    if not isinstance(test_obj, dict):
        return {
            "_test_index": i,
            "test_id": f"test_{i}",
            "status": "invalid_config",
            "error": "test entry is not a mapping",
        }
    test_id = str(test_obj.get("id", f"test_{i}"))

    try:
        predict_io = PredictIO(
            bam=as_path(test_obj.get("bam"), f"tests[{i}].bam"),
            gtf=as_path(test_obj.get("gtf"), f"tests[{i}].gtf"),
            model_dir=(
                Path(str(test_obj["model_dir"])) if test_obj.get("model_dir") else job.model_dir
            ),
            outdir=(
                Path(str(test_obj["outdir"]))
                if test_obj.get("outdir")
                else (job.bench_outdir / "tests" / test_id)
            ),
            config_file=(
                Path(str(test_obj["config"])) if test_obj.get("config") else job.train_config
            ),
        )
    except ValueError as exc:
        return {
            "_test_index": i,
            "test_id": test_id,
            "status": "invalid_config",
            "error": str(exc),
        }

    extra = {"test_id": test_id, "test_index": str(i)}
    t0 = __import__("time").perf_counter()
    log_ts(f"[telos_repro] benchmark {i}/{job.n_tests}: predict {test_id} start")
    predict_code = run_predict(predict_io)
    predict_elapsed = elapsed_s(t0)
    log_ts(
        f"[telos_repro] benchmark {i}/{job.n_tests}: predict {test_id} end "
        f"elapsed_s={predict_elapsed} code={predict_code}"
    )

    ranked_rf = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_RF_TSV
    ranked_xgb = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_XGB_TSV
    sites_scored = predict_io.outdir / "predictions" / "sites.scored.tsv"
    aid = test_obj.get("assembler_id")
    row: dict[str, Any] = {
        "_test_index": i,
        "test_id": test_id,
        "assembler_id": str(aid).strip() if aid is not None and str(aid).strip() else "",
        "predict_code": str(predict_code),
        "status": "ok" if predict_code == 0 else "predict_failed",
        "error": "",
        "predict_outdir": str(predict_io.outdir),
        "ranked_rf_tsv": str(ranked_rf) if ranked_rf.exists() else "",
        "ranked_xgb_tsv": str(ranked_xgb) if ranked_xgb.exists() else "",
        "predict_elapsed_s": predict_elapsed,
    }

    analysis_elapsed: float | None = None
    if predict_code == 0 and job.analyze and test_obj.get("tmap"):
        ref_gtf_raw = test_obj.get("ref_gtf")
        curves_dir = predict_io.outdir / "reports" / "curves"
        t_an = __import__("time").perf_counter()
        try:
            tmap_path = Path(str(test_obj["tmap"]))
            assembly_gtf = Path(str(test_obj["gtf"]))
            for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                m = stage2_aupr_from_ranked_vs_tmap_and_cov(rp, tmap_path, assembly_gtf)
                row[f"stage2_test_aupr_{suf}"] = m["aupr_model"]
                row[f"stage2_test_aupr_novel_{suf}"] = m["aupr_model_novel"]
                row["stage2_test_n_eval_tx"] = m["n_joined"]
                row["stage2_test_n_novel_pos_tx"] = m["n_pos_novel"]
                if "stage2_test_aupr_baseline" not in row:
                    row["stage2_test_aupr_baseline"] = m["aupr_baseline"]
                if "stage2_test_aupr_novel_baseline" not in row:
                    row["stage2_test_aupr_novel_baseline"] = m["aupr_baseline_novel"]
            write_stage2_curve_points(
                ranked_rf_tsv=ranked_rf,
                ranked_xgb_tsv=ranked_xgb,
                tmap_path=tmap_path,
                assembly_gtf=assembly_gtf,
                out_path=curves_dir / "stage2_transcript_curve_points.tsv",
            )
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
                    include_curve_points=True,
                )
                curve_points = st1.pop("_curve_points", {}) if isinstance(st1, dict) else {}
                row.update(st1)
                tss_df = curve_points.get("tss")
                tes_df = curve_points.get("tes")
                if isinstance(tss_df, pd.DataFrame) and not tss_df.empty:
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    tss_df.to_csv(curves_dir / "stage1_tss_curve_points.tsv", sep="\t", index=False)
                if isinstance(tes_df, pd.DataFrame) and not tes_df.empty:
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    tes_df.to_csv(curves_dir / "stage1_tes_curve_points.tsv", sep="\t", index=False)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                row["error"] = f"stage1_test_aupr_failed: {exc}"

        if job.sklearn_bundle_tmap:
            try:
                tmap_path = Path(str(test_obj["tmap"]))
                for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                    m = sklearn_metrics_ranked_vs_bundle_tmap(rp, tmap_path)
                    row.update(suffix_keys(m, suf))
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                row["analysis_error"] = str(exc)

        if job.pr_vs_baseline:
            t_pr = __import__("time").perf_counter()
            try:
                pred_cfg = load_mapping_config(predict_io.config_file)
                validate_stage_config(pred_cfg)
            except ValueError as exc:
                row["transcript_pr_error"] = f"stage config invalid: {exc}"
                pred_cfg = {}
            pr_reports = predict_io.outdir / "reports" / "pr"
            measure = str(job.pr_block.get("measure", "cov"))
            gffcompare_ex = job.pr_block.get("gffcompare_bin")
            gffcompare_bin = str(gffcompare_ex).strip() if gffcompare_ex else None
            split_pol = str(get_nested(pred_cfg, ["stage1", "training", "split_policy"], "chr1-10"))
            pr_errs: list[str] = []
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
            elif not row.get("transcript_pr_error"):
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
                            plot=job.pr_plot_effective,
                            save_pr_tables=job.pr_save_tables_effective,
                            chromosomes_path=job.pr_chromosomes_path,
                            filter_validation_chroms=job.pr_filter_val_chroms,
                            autosome_train_range=autosome_train_range,
                            ephemeral_workdir=job.ephemeral_pr,
                        )
                        merge_transcript_pr_into_row(row, pr_row, suf)
                        if "transcript_pr_auc_baseline" not in row:
                            row["transcript_pr_auc_baseline"] = pr_row.get("transcript_pr_auc_baseline")
                    pr_bits = []
                    for suf, lab in (("rf", "RF"), ("xgb", "XGB")):
                        v = row.get(f"transcript_pr_auc_model_{suf}")
                        if isinstance(v, (int, float)):
                            pr_bits.append(f"{lab} transcript_pr_auc={float(v):.2f}")
                    log_ts(
                        f"[telos_repro] benchmark transcript PR {test_id}: "
                        + (", ".join(pr_bits) if pr_bits else "no AUC")
                        + (" (ephemeral workdir)" if job.ephemeral_pr else f" dir={pr_reports}")
                    )
                except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as exc:
                    row["transcript_pr_error"] = str(exc)
            row["transcript_pr_elapsed_s"] = elapsed_s(t_pr)

        analysis_elapsed = elapsed_s(t_an)
        row["analysis_elapsed_s"] = analysis_elapsed

    if job.timing_csv is not None:
        from telos_repro.benchmark.timing import append_timing_row

        append_timing_row(
            job.timing_csv,
            {
                **extra,
                "phase": "test",
                "started_at": utc_now_iso(),
                "elapsed_s": str(elapsed_s(t0)),
                "predict_elapsed_s": str(predict_elapsed),
                "status": str(row.get("status", "")),
            },
        )

    round_float_metrics_in_row(row)
    return row
