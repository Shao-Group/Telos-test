from __future__ import annotations

import csv
import json
import numbers
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from telos_v2.commands.predict import run_predict
from telos_v2.commands.train import run_train
from telos_v2.config_loader import get_nested, load_mapping_config
from telos_v2.config_models import PredictIO, TrainIO
from telos_v2.config_models import BenchmarkIO
from telos_v2.models.chrom_split import parse_split_policy
from telos_v2.backends.gtfcuff import GtfcuffError, resolve_gtfcuff_binary
from telos_v2.backends.gtfformat import GtfformatError, resolve_gtfformat_binary
from telos_v2.evaluation.benchmark_sklearn_metrics import (
    sklearn_metrics_ranked_vs_bundle_tmap,
    suffix_keys,
)
from telos_v2.evaluation.transcript_pr_pipeline import run_transcript_pr_benchmark_gtfcuff
from telos_v2.labels.site_labels import label_sites_by_proximity, reference_sites_from_gtf
from telos_v2.models import TRANSCRIPTS_RANKED_RF_TSV, TRANSCRIPTS_RANKED_XGB_TSV
from telos_v2.reporting.run_manifest import build_run_manifest, write_run_manifest
from telos_v2.reporting.summary import write_benchmark_summary
from telos_v2.validation.preflight import (
    PreflightError,
    ensure_run_layout,
    run_preflight_benchmark,
)


def _as_path(v: Any, name: str) -> Path:
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required benchmark field: {name}")
    return Path(str(v))


def _round_float_metrics_in_row(row: dict[str, Any], *, ndigits: int = 2) -> None:
    """Round non-integral numeric metrics for tabular/JSON output (keeps bools, ints, strings)."""
    for k, v in list(row.items()):
        if isinstance(v, bool):
            continue
        if isinstance(v, numbers.Integral):
            continue
        if isinstance(v, numbers.Real):
            row[k] = round(float(v), ndigits)


def _write_benchmark_summary_json(
    reports_dir: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    partial: bool,
) -> Path:
    """Write combined train + tests summary JSON (updated after each test when ``partial``)."""
    path = reports_dir / "benchmark_summary.json"
    payload: dict[str, Any] = {"train": train_rows, "tests": test_rows}
    if partial:
        payload["partial"] = True
        payload["note"] = (
            "Run in progress or ended before finalization; "
            "benchmark_summary.csv and benchmark_summary.md are written only when all tests finish."
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _merge_pr_side(row: dict[str, Any], pr: dict[str, Any], suf: str) -> None:
    for k, v in pr.items():
        if k == "pr_reports_dir":
            row[k] = v
        elif k.startswith("pr_"):
            row[f"{k}_{suf}"] = v
        else:
            row[k] = v


def _safe_aupr(y_true: pd.Series, scores: pd.Series) -> float | None:
    yt = pd.to_numeric(y_true, errors="coerce")
    sc = pd.to_numeric(scores, errors="coerce")
    mask = yt.notna() & sc.notna()
    if not bool(mask.any()):
        return None
    y = yt[mask].astype(int)
    s = sc[mask].astype(float)
    if y.nunique() < 2:
        return None
    return float(average_precision_score(y, s))


def _evaluate_stage1_test_aupr(
    sites_scored_tsv: Path,
    cov_tsv: Path,
    ref_gtf: Path,
    *,
    tolerance_bp: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    df = pd.read_csv(sites_scored_tsv, sep="\t")
    if df.empty:
        return out
    needed = {"site_type", "chrom", "position", "strand", "p_site_rf", "p_site_xgb"}
    if not needed.issubset(df.columns):
        return out
    ref_df = reference_sites_from_gtf(ref_gtf)
    if ref_df.empty:
        return out
    cov = pd.read_csv(cov_tsv, sep="\t", dtype={"tss_chrom": str, "tes_chrom": str})

    for st in ("TSS", "TES"):
        sub = df[df["site_type"].astype(str).str.upper() == st].copy()
        if sub.empty:
            continue
        if st == "TSS":
            csub = cov[["tss_chrom", "tss_pos", "coverage"]].copy()
            if len(csub) > csub[["tss_chrom", "tss_pos"]].drop_duplicates().shape[0]:
                csub = csub.groupby(["tss_chrom", "tss_pos"], as_index=False)["coverage"].mean()
            sub = sub.merge(
                csub,
                left_on=["chrom", "position"],
                right_on=["tss_chrom", "tss_pos"],
                how="inner",
            )
        else:
            csub = cov[["tes_chrom", "tes_pos", "coverage"]].copy()
            if len(csub) > csub[["tes_chrom", "tes_pos"]].drop_duplicates().shape[0]:
                csub = csub.groupby(["tes_chrom", "tes_pos"], as_index=False)["coverage"].mean()
            sub = sub.merge(
                csub,
                left_on=["chrom", "position"],
                right_on=["tes_chrom", "tes_pos"],
                how="inner",
            )
        if sub.empty:
            continue
        labels = label_sites_by_proximity(sub, ref_df, st, tolerance_bp)
        out[f"stage1_test_aupr_{st.lower()}_rf"] = _safe_aupr(labels, sub["p_site_rf"])
        out[f"stage1_test_aupr_{st.lower()}_xgb"] = _safe_aupr(labels, sub["p_site_xgb"])
        out[f"stage1_test_aupr_{st.lower()}_baseline"] = _safe_aupr(labels, sub["coverage"])
    return out


def _stub_test_row(
    test_id: str,
    *,
    status: str,
    error: str,
    train_metric_cols: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "test_id": test_id,
        "assembler_id": "",
        "predict_code": "2",
        "status": status,
        "error": error,
        "predict_outdir": "",
        "ranked_rf_tsv": "",
        "ranked_xgb_tsv": "",
        "filtered_gtf_rf": "",
        "filtered_gtf_xgb": "",
    }
    _round_float_metrics_in_row(row)
    return row


def run_benchmark(cfg: BenchmarkIO) -> int:
    """Train once, predict on many test datasets, and summarize metrics."""
    try:
        run_preflight_benchmark(cfg.config)
    except PreflightError as exc:
        print(f"[telos_v2] preflight failed: {exc}")
        return 2

    layout = ensure_run_layout(cfg.outdir)
    try:
        bench_cfg = load_mapping_config(cfg.config)
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

    manifest = build_run_manifest(
        command="benchmark",
        args_dict={
            "config": str(cfg.config),
            "outdir": str(cfg.outdir),
            "n_tests": len(tests),
        },
    )
    manifest_path = write_run_manifest(manifest, layout.reports_dir)
    root = layout.root.resolve()
    print(f"[telos_v2] benchmark output root: {root}")
    print(f"[telos_v2]   manifest + summaries -> {layout.reports_dir.resolve()}/")
    print(
        f"[telos_v2]   per-test outputs -> {root}/tests/<id>/ "
        f"(train may use train.outdir from YAML under this tree or elsewhere)"
    )

    stop_on_error = bool(get_nested(bench_cfg, ["execution", "stop_on_error"], False))
    train_rows: list[dict[str, Any]] = []
    train_mode = str(get_nested(train_obj, ["mode"], "run")).strip().lower()
    train_config = Path(str(train_obj["config"])) if train_obj.get("config") else None
    train_out = (
        Path(str(train_obj["outdir"]))
        if train_obj.get("outdir")
        else (cfg.outdir / "train")
    )

    model_dir: Path
    if train_mode == "skip":
        try:
            model_dir = _as_path(train_obj.get("model_dir"), "train.model_dir")
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
        (layout.reports_dir / "benchmark_train.json").write_text(
            json.dumps(train_rows, indent=2),
            encoding="utf-8",
        )
        print(f"[telos_v2]   train skipped; using models -> {model_dir.resolve()}/")
    else:
        try:
            train_io = TrainIO(
                bam=_as_path(train_obj.get("bam"), "train.bam"),
                gtf=_as_path(train_obj.get("gtf"), "train.gtf"),
                ref_gtf=_as_path(train_obj.get("ref_gtf"), "train.ref_gtf"),
                tmap=_as_path(train_obj.get("tmap"), "train.tmap"),
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
        train_summary_path = layout.reports_dir / "benchmark_train.json"
        train_summary_path.write_text(json.dumps(train_rows, indent=2), encoding="utf-8")
        if train_code != 0:
            print(f"[telos_v2] benchmark aborted: training failed (code={train_code})")
            print(f"  train_summary={train_summary_path}")
            return 1
        print(f"[telos_v2]   training run -> {train_out.resolve()}/")

    train_metric_cols: dict[str, Any] = {}
    for tr in train_rows:
        _round_float_metrics_in_row(tr)

    rows: list[dict[str, Any]] = []
    analyze = bool(get_nested(bench_cfg, ["analysis", "enabled"], True))
    pr_vs_baseline = bool(get_nested(bench_cfg, ["analysis", "pr_vs_baseline", "enabled"], True))
    pr_plot = bool(get_nested(bench_cfg, ["analysis", "pr_vs_baseline", "plot"], True))
    pr_block = get_nested(bench_cfg, ["analysis", "pr_vs_baseline"], {}) or {}
    pr_save_tables = bool(pr_block.get("save_pr_tables", False))
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

    # Legacy transcript AUPR uses gtfcuff; sklearn×bundle-tmap is redundant and misleading when PR runs.
    sklearn_bundle_tmap = analyze and not pr_vs_baseline

    def _snapshot_tests_partial() -> Path:
        p = _write_benchmark_summary_json(layout.reports_dir, train_rows, rows, partial=True)
        print(f"[telos_v2]   benchmark_summary.json (partial) -> {p}")
        return p

    _write_benchmark_summary_json(layout.reports_dir, train_rows, [], partial=True)
    print(f"[telos_v2]   benchmark_summary.json (partial) -> {layout.reports_dir / 'benchmark_summary.json'}")

    for i, test_obj in enumerate(tests, start=1):
        if not isinstance(test_obj, dict):
            rows.append(
                _stub_test_row(
                    f"test_{i}",
                    status="invalid_config",
                    error="test entry is not a mapping",
                    train_metric_cols=train_metric_cols,
                )
            )
            _snapshot_tests_partial()
            if stop_on_error:
                break
            continue

        test_id = str(test_obj.get("id", f"test_{i}"))
        try:
            predict_io = PredictIO(
                bam=_as_path(test_obj.get("bam"), f"tests[{i}].bam"),
                gtf=_as_path(test_obj.get("gtf"), f"tests[{i}].gtf"),
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
                _stub_test_row(
                    test_id,
                    status="invalid_config",
                    error=str(exc),
                    train_metric_cols=train_metric_cols,
                )
            )
            _snapshot_tests_partial()
            if stop_on_error:
                break
            continue

        print(f"[telos_v2] benchmark {i}/{len(tests)}: predict {test_id}")
        predict_code = run_predict(predict_io)
        ranked_rf = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_RF_TSV
        ranked_xgb = predict_io.outdir / "predictions" / TRANSCRIPTS_RANKED_XGB_TSV
        sites_scored = predict_io.outdir / "predictions" / "sites.scored.tsv"
        filtered_rf = predict_io.outdir / "filtered" / "transcripts.filtered.rf.gtf"
        filtered_xgb = predict_io.outdir / "filtered" / "transcripts.filtered.xgb.gtf"
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
            "filtered_gtf_rf": str(filtered_rf) if filtered_rf.exists() else "",
            "filtered_gtf_xgb": str(filtered_xgb) if filtered_xgb.exists() else "",
        }
        if predict_code == 0 and test_obj.get("tmap"):
            ref_gtf_raw = test_obj.get("ref_gtf")
            if ref_gtf_raw and sites_scored.is_file():
                pred_cfg = load_mapping_config(predict_io.config_file)
                tol = int(get_nested(pred_cfg, ["stage1", "training", "site_label_tolerance_bp"], 50))
                cov_name = str(get_nested(pred_cfg, ["rnaseqtools", "cov_tsv_name"], "cov.tsv"))
                cov_tsv = predict_io.outdir / "predictions" / cov_name
                try:
                    st1 = _evaluate_stage1_test_aupr(
                        sites_scored,
                        cov_tsv,
                        Path(str(ref_gtf_raw)),
                        tolerance_bp=tol,
                    )
                    row.update(st1)
                except Exception as exc:
                    row["error"] = f"stage1_test_aupr_failed: {exc}"

            if sklearn_bundle_tmap:
                try:
                    tmap_path = Path(str(test_obj["tmap"]))
                    for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                        m = sklearn_metrics_ranked_vs_bundle_tmap(rp, tmap_path)
                        row.update(suffix_keys(m, suf))
                except Exception as exc:
                    row["analysis_error"] = str(exc)

            if pr_vs_baseline:
                pred_cfg = load_mapping_config(predict_io.config_file)
                pr_reports = predict_io.outdir / "reports" / "pr"
                measure = str(pr_block.get("measure", "cov"))
                gffcompare_ex = pr_block.get("gffcompare_bin")
                gffcompare_bin = str(gffcompare_ex).strip() if gffcompare_ex else None
                gtfcuff_ex = pr_block.get("gtfcuff_bin")
                gtfformat_ex = get_nested(pred_cfg, ["rnaseqtools", "gtfformat_bin"], None)
                split_pol = str(
                    get_nested(pred_cfg, ["stage1", "training", "split_policy"], "chr1-10")
                )
                pr_errs: list[str] = []
                ref_gtf_raw = test_obj.get("ref_gtf")
                if not ref_gtf_raw:
                    pr_errs.append(
                        "benchmark PR requires test.ref_gtf (reference annotation for gffcompare). "
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
                    row["pr_error"] = "; ".join(pr_errs)
                else:
                    try:
                        gtfformat_bin = resolve_gtfformat_binary(
                            Path(str(gtfformat_ex))
                            if gtfformat_ex is not None and str(gtfformat_ex).strip()
                            else None
                        )
                        gtfcuff_bin = resolve_gtfcuff_binary(
                            Path(str(gtfcuff_ex))
                            if gtfcuff_ex is not None and str(gtfcuff_ex).strip()
                            else None
                        )
                    except (GtfformatError, GtfcuffError) as exc:
                        row["pr_error"] = str(exc)

                    if not row.get("pr_error"):
                        assembly_gtf = Path(str(test_obj["gtf"]))
                        ref_gtf = Path(str(ref_gtf_raw))
                        try:
                            row["pr_reports_dir"] = str(pr_reports)
                            for suf, rp in (("rf", ranked_rf), ("xgb", ranked_xgb)):
                                pr_row = run_transcript_pr_benchmark_gtfcuff(
                                    assembly_gtf=assembly_gtf,
                                    ref_gtf=ref_gtf,
                                    ranked_tsv=rp,
                                    reports_pr_dir=pr_reports,
                                    work_rel=f"gtfcuff_{test_id}_{suf}",
                                    prefix=f"transcript_pr_{suf}",
                                    gtfformat_bin=gtfformat_bin,
                                    gtfcuff_bin=gtfcuff_bin,
                                    gffcompare_bin=gffcompare_bin,
                                    measure=measure,
                                    score_col="pred_prob",
                                    plot=pr_plot,
                                    plot_filename=f"transcript_pr_{suf}.png",
                                    chromosomes_file=pr_chromosomes_path,
                                    filter_validation_chroms=pr_filter_val_chroms,
                                    autosome_train_range=autosome_train_range,
                                    save_pr_tables=pr_save_tables,
                                )
                                _merge_pr_side(row, pr_row, suf)
                            row["stage2_test_aupr_rf"] = row.get("pr_aupr_model_rf")
                            row["stage2_test_aupr_xgb"] = row.get("pr_aupr_model_xgb")
                            row["stage2_test_aupr_baseline_rf"] = row.get("pr_aupr_baseline_rf")
                            row["stage2_test_aupr_baseline_xgb"] = row.get("pr_aupr_baseline_xgb")
                            pr_bits = []
                            for suf, lab in (("rf", "RF"), ("xgb", "XGB")):
                                v = row.get(f"pr_aupr_model_{suf}")
                                if isinstance(v, (int, float)):
                                    pr_bits.append(f"{lab} gtfcuff_auc={float(v):.2f}")
                            print(
                                f"[telos_v2] benchmark PR {test_id}: "
                                + (", ".join(pr_bits) if pr_bits else "no AUC")
                                + f" dir={pr_reports}"
                            )
                        except Exception as exc:
                            row["pr_error"] = str(exc)

        _round_float_metrics_in_row(row)
        rows.append(row)
        _snapshot_tests_partial()
        if predict_code != 0 and stop_on_error:
            break

    summary_csv = layout.reports_dir / "benchmark_summary.csv"
    summary_json = _write_benchmark_summary_json(
        layout.reports_dir, train_rows, rows, partial=False
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        all_keys: set[str] = set()
        for r in rows:
            all_keys.update(r.keys())
        ordered_keys = [
            "test_id",
            "assembler_id",
            "status",
            "predict_code",
            "stage1_test_aupr_tss_rf",
            "stage1_test_aupr_tss_xgb",
            "stage1_test_aupr_tss_baseline",
            "stage1_test_aupr_tes_rf",
            "stage1_test_aupr_tes_xgb",
            "stage1_test_aupr_tes_baseline",
            "stage2_test_aupr_rf",
            "stage2_test_aupr_xgb",
            "stage2_test_aupr_baseline_rf",
            "stage2_test_aupr_baseline_xgb",
            "error",
        ]
        # Keep benchmark_summary.csv concise and stable for downstream tables.
        fields = [k for k in ordered_keys if k in all_keys]
        compact_rows: list[dict[str, Any]] = []
        for r in rows:
            compact_rows.append({k: r.get(k, "") for k in fields})
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(compact_rows)

    bench_summary_md = write_benchmark_summary(
        layout.reports_dir,
        config_path=cfg.config,
        outdir=cfg.outdir,
        manifest_path=manifest_path,
        summary_tsv=summary_csv,
        summary_json=summary_json,
        train_rows=train_rows,
        test_rows=rows,
    )

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print("[telos_v2] benchmark complete")
    print(f"  config={cfg.config}")
    print(f"  manifest={manifest_path}")
    print(f"  benchmark_summary_json={summary_json.resolve()}")
    print(f"  benchmark_summary_csv={summary_csv.resolve()}")
    print(f"  benchmark_summary_md={bench_summary_md}")
    print(f"  completed_ok={ok}/{len(rows)}")
    return 0 if ok == len(rows) else 1
