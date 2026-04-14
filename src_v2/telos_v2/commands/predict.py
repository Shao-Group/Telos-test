from __future__ import annotations

from pathlib import Path

import pandas as pd

from telos_v2.backends.gtfformat import GtfformatError
from telos_v2.candidates.load import load_candidates, run_transcript_cov_table
from telos_v2.candidates.extract import write_candidates_tsv
from telos_v2.config_loader import get_nested, load_mapping_config
from telos_v2.config_models import PredictIO
from telos_v2.features.stage1 import Stage1FeatureConfig, compute_stage1_features, write_feature_tsv
from telos_v2.models import (
    SITE_PROB_COLUMN_RF,
    SITE_PROB_COLUMN_XGB,
    STAGE1_BACKEND_RF,
    STAGE1_BACKEND_XGB,
    STAGE1_BACKENDS,
    stage2_model_joblib_for_backend,
)
from telos_v2.models.stage1_predict import score_stage1_dataframe, write_sites_scored_tsv
from telos_v2.models.stage2_predict import run_stage2_predict
from telos_v2.models.stage2_train import build_stage2_inference_frame
from telos_v2.postprocess.filter_gtf import filter_gtf_by_transcript_scores
from telos_v2.reporting.run_manifest import build_run_manifest, write_run_manifest
from telos_v2.reporting.summary import write_predict_summaries
from telos_v2.validation.preflight import (
    PreflightError,
    ensure_run_layout,
    run_preflight_predict,
)


def run_predict(cfg: PredictIO) -> int:
    """Run Stage I + II inference; write site and transcript scores."""
    cfg_map = load_mapping_config(cfg.config_file)
    save_intermediates = bool(get_nested(cfg_map, ["outputs", "save_intermediates"], False))
    try:
        run_preflight_predict(cfg.bam, cfg.gtf, cfg.model_dir, cfg_map)
    except PreflightError as exc:
        print(f"[telos_v2] preflight failed: {exc}")
        return 2

    layout = ensure_run_layout(cfg.outdir, save_intermediates=save_intermediates)
    feat_cfg = Stage1FeatureConfig(
        window_size=int(get_nested(cfg_map, ["stage1", "feature_extraction", "window_size"], 100)),
        density_window=int(get_nested(cfg_map, ["stage1", "feature_extraction", "density_window"], 100)),
        coverage_window=int(get_nested(cfg_map, ["stage1", "feature_extraction", "coverage_window"], 100)),
        soft_clip_window=int(get_nested(cfg_map, ["stage1", "feature_extraction", "soft_clip_window"], 10)),
        min_mapq=int(get_nested(cfg_map, ["stage1", "feature_extraction", "min_mapq"], 10)),
        splice_site_window=int(get_nested(cfg_map, ["stage1", "feature_extraction", "splice_site_window"], 300)),
        gradient_analysis_range=int(
            get_nested(cfg_map, ["stage1", "feature_extraction", "gradient_analysis_range"], 100)
        ),
        extended_window=int(get_nested(cfg_map, ["stage1", "feature_extraction", "extended_window"], 1000)),
        k_values=tuple(get_nested(cfg_map, ["stage1", "features", "k_values"], [3])),
    )
    stage1_parallel_yaml = bool(get_nested(cfg_map, ["stage1", "feature_extraction", "parallel"], True))
    stage1_parallel = stage1_parallel_yaml and not cfg.stage1_no_parallel
    stage1_parallel_min = int(
        get_nested(cfg_map, ["stage1", "feature_extraction", "parallel_min_sites"], 50)
    )
    stage1_n_workers_yaml = get_nested(cfg_map, ["stage1", "feature_extraction", "n_workers"], None)
    stage1_n_workers = cfg.stage1_n_workers if cfg.stage1_n_workers is not None else stage1_n_workers_yaml
    cand_src = str(get_nested(cfg_map, ["candidates", "source"], "gtfformat"))
    cov_name = str(get_nested(cfg_map, ["rnaseqtools", "cov_tsv_name"], "cov.tsv"))
    gtff_bin_cfg = get_nested(cfg_map, ["rnaseqtools", "gtfformat_bin"], "")
    manifest = build_run_manifest(
        command="predict",
        args_dict={
            "bam": str(cfg.bam),
            "gtf": str(cfg.gtf),
            "model_dir": str(cfg.model_dir),
            "outdir": str(cfg.outdir),
            "config_file": str(cfg.config_file) if cfg.config_file else "",
            "candidates_source": cand_src,
            "rnaseqtools": {
                "gtfformat_bin": str(gtff_bin_cfg) if gtff_bin_cfg else "",
                "run_get_cov": bool(get_nested(cfg_map, ["rnaseqtools", "run_get_cov"], True)),
                "cov_tsv_name": cov_name,
            },
            "stage1_feature_config": {
                "window_size": feat_cfg.window_size,
                "density_window": feat_cfg.density_window,
                "coverage_window": feat_cfg.coverage_window,
                "soft_clip_window": feat_cfg.soft_clip_window,
                "min_mapq": feat_cfg.min_mapq,
                "splice_site_window": feat_cfg.splice_site_window,
                "gradient_analysis_range": feat_cfg.gradient_analysis_range,
                "extended_window": feat_cfg.extended_window,
                "k_values": list(feat_cfg.k_values),
            },
            "stage1_feature_parallel": {
                "parallel": stage1_parallel,
                "parallel_min_sites": stage1_parallel_min,
                "n_workers": stage1_n_workers,
                "cli_no_parallel": cfg.stage1_no_parallel,
                "cli_n_workers": cfg.stage1_n_workers,
            },
            "stage1_training": {
                "site_label_tolerance_bp": int(
                    get_nested(cfg_map, ["stage1", "training", "site_label_tolerance_bp"], 50)
                ),
                "split_policy": str(get_nested(cfg_map, ["stage1", "training", "split_policy"], "chr1-10")),
                "random_state": int(get_nested(cfg_map, ["stage1", "training", "random_state"], 42)),
            },
            "stage2_filtering": {
                "transcript_prob_threshold": float(
                    get_nested(cfg_map, ["stage2", "filtering", "transcript_prob_threshold"], 0.5)
                )
            },
            "outputs": {"save_intermediates": save_intermediates},
        },
    )
    manifest_path = write_run_manifest(manifest, layout.reports_dir)

    try:
        candidates = load_candidates(cfg_map, cfg.gtf)
    except ValueError as exc:
        print(f"[telos_v2] candidates config error: {exc}")
        return 2

    if not candidates:
        print(
            "[telos_v2] no candidates from configured source. "
            "For gtf_transcript, check transcript features; for gtfformat, ensure TSSTES emits sites."
        )
        return 3

    candidate_path: Path | None = None
    if save_intermediates:
        candidate_path = layout.predictions_dir / "candidates.tsv"
        write_candidates_tsv(candidates, candidate_path)

    cov_path = layout.predictions_dir / cov_name
    try:
        run_transcript_cov_table(cfg_map, cfg.gtf, cov_path)
    except GtfformatError as exc:
        print(f"[telos_v2] get-cov failed: {exc}")
        return 2

    rows = compute_stage1_features(
        cfg.bam,
        candidates,
        feat_cfg,
        parallel=stage1_parallel,
        parallel_min_sites=stage1_parallel_min,
        n_workers=stage1_n_workers,
    )
    tss_path: Path | None = None
    tes_path: Path | None = None
    if save_intermediates:
        tss_rows = [r for r in rows if str(r["site_type"]).upper() == "TSS"]
        tes_rows = [r for r in rows if str(r["site_type"]).upper() == "TES"]
        tss_path = layout.predictions_dir / "stage1_tss_features.tsv"
        tes_path = layout.predictions_dir / "stage1_tes_features.tsv"
        write_feature_tsv(tss_rows, tss_path)
        write_feature_tsv(tes_rows, tes_path)

    df_all = pd.DataFrame(rows)
    try:
        sites_scored = score_stage1_dataframe(df_all, cfg.model_dir)
    except FileNotFoundError as exc:
        print(f"[telos_v2] predict: {exc}")
        return 2
    sites_path = layout.predictions_dir / "sites.scored.tsv"
    write_sites_scored_tsv(sites_scored, sites_path)

    prob_cols = {STAGE1_BACKEND_RF: SITE_PROB_COLUMN_RF, STAGE1_BACKEND_XGB: SITE_PROB_COLUMN_XGB}
    ranked_paths: dict[str, Path] = {}
    filtered_paths: dict[str, Path] = {}
    kept: dict[str, tuple[int, int]] = {}

    filter_threshold = float(
        get_nested(cfg_map, ["stage2", "filtering", "transcript_prob_threshold"], 0.5)
    )

    for backend in STAGE1_BACKENDS:
        prob_col = prob_cols[backend]
        try:
            df_stage2 = build_stage2_inference_frame(
                cov_path, df_all, sites_scored, site_prob_column=prob_col
            )
        except Exception as exc:
            print(f"[telos_v2] Stage II merge failed ({backend}): {exc}")
            return 2
        if df_stage2.empty:
            print(f"[telos_v2] Stage II merged table is empty (cov × sites, backend={backend}).")
            return 2
        try:
            ranked_paths[backend] = run_stage2_predict(
                df_stage2, cfg.model_dir, layout.predictions_dir, stage1_backend_tag=backend
            )
        except Exception as exc:
            print(f"[telos_v2] Stage II predict failed ({backend}): {exc}")
            return 2

        filtered_gtf_path = layout.filtered_dir / f"transcripts.filtered.{backend}.gtf"
        try:
            k, t = filter_gtf_by_transcript_scores(
                cfg.gtf,
                ranked_paths[backend],
                filtered_gtf_path,
                filter_threshold,
            )
            kept[backend] = (k, t)
            filtered_paths[backend] = filtered_gtf_path
        except Exception as exc:
            print(f"[telos_v2] filtered GTF generation failed ({backend}): {exc}")
            return 2

    k_rf, t_rf = kept[STAGE1_BACKEND_RF]
    k_xgb, t_xgb = kept[STAGE1_BACKEND_XGB]

    def _count_rows(path: Path) -> int | None:
        try:
            with path.open(encoding="utf-8") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except OSError:
            return None

    n_rf = _count_rows(ranked_paths[STAGE1_BACKEND_RF])
    n_xgb = _count_rows(ranked_paths[STAGE1_BACKEND_XGB])
    n_ranked_note = (
        f"rf={n_rf}, xgb={n_xgb}" if n_rf is not None and n_xgb is not None else None
    )

    pred_paths: dict[str, Path | str] = {
        "manifest": manifest_path,
        "bam": cfg.bam,
        "gtf": cfg.gtf,
        "model_dir": cfg.model_dir,
        "outdir": cfg.outdir,
        "sites_scored": sites_path,
        "transcripts_ranked_rf": ranked_paths[STAGE1_BACKEND_RF],
        "transcripts_ranked_xgb": ranked_paths[STAGE1_BACKEND_XGB],
        "filtered_gtf_rf": filtered_paths[STAGE1_BACKEND_RF],
        "filtered_gtf_xgb": filtered_paths[STAGE1_BACKEND_XGB],
    }
    summary_md, predict_summary_md = write_predict_summaries(
        layout.reports_dir,
        manifest_path=manifest_path,
        paths=pred_paths,
        kept_tx=k_rf,
        total_tx=t_rf,
        filter_threshold=filter_threshold,
        n_ranked_transcripts=n_ranked_note,
        kept_tx_xgb=k_xgb,
        total_tx_xgb=t_xgb,
    )

    print("[telos_v2] predict complete (Stage I + Stage II)")
    print(f"  bam={cfg.bam}")
    print(f"  gtf={cfg.gtf}")
    print(f"  model_dir={cfg.model_dir}")
    print(f"  outdir={cfg.outdir}")
    print(f"  manifest={manifest_path}")
    if candidate_path is not None:
        print(f"  candidates={candidate_path}")
    if get_nested(cfg_map, ["rnaseqtools", "run_get_cov"], True):
        print(f"  cov_tsv={cov_path}")
    if tss_path is not None and tes_path is not None:
        print(f"  stage1_tss={tss_path}")
        print(f"  stage1_tes={tes_path}")
    print(f"  sites_scored={sites_path}")
    for b in STAGE1_BACKENDS:
        print(f"  stage2_model_{b}={cfg.model_dir / stage2_model_joblib_for_backend(b)}")
        print(f"  transcripts_ranked_{b}={ranked_paths[b]}")
        print(f"  filtered_gtf_{b}={filtered_paths[b]}")
    print(
        f"  filter: rf kept={k_rf}/{t_rf}, xgb kept={k_xgb}/{t_xgb} (threshold={filter_threshold})"
    )
    print(f"  summary_md={summary_md}")
    print(f"  predict_summary_md={predict_summary_md}")
    return 0
