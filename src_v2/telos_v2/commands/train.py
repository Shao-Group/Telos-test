from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from telos_v2.backends.gtfformat import GtfformatError
from telos_v2.candidates.load import load_candidates, run_transcript_cov_table
from telos_v2.candidates.extract import write_candidates_tsv
from telos_v2.config_loader import get_nested, load_mapping_config
from telos_v2.config_models import TrainIO
from telos_v2.features.stage1 import Stage1FeatureConfig, compute_stage1_features, write_feature_tsv
from telos_v2.labels.site_labels import label_sites_by_proximity, reference_sites_from_gtf
from telos_v2.models import (
    SITE_PROB_COLUMN_RF,
    SITE_PROB_COLUMN_XGB,
    STAGE1_BACKEND_RF,
    STAGE1_BACKEND_XGB,
    STAGE1_BACKENDS,
    stage1_bundle_path,
    stage2_model_joblib_for_backend,
    transcripts_ranked_tsv_for_backend,
)
from telos_v2.models.chrom_split import parse_split_policy
from telos_v2.models.stage1_predict import score_stage1_dataframe, write_sites_scored_tsv
from telos_v2.models.stage1_train import (
    save_stage1_bundle,
    train_stage1_site_classifier,
    write_train_metrics,
)
from telos_v2.models.stage2_train import build_stage2_training_frame, train_and_save_stage2
from telos_v2.reporting.run_manifest import build_run_manifest, write_run_manifest
from telos_v2.reporting.summary import write_train_summary
from telos_v2.validation.preflight import (
    PreflightError,
    ensure_run_layout,
    run_preflight_train,
)


def run_train(cfg: TrainIO) -> int:
    """Train Stage I + II; write models, primary predictions, and manifest."""
    cfg_map = load_mapping_config(cfg.config_file)
    save_intermediates = bool(get_nested(cfg_map, ["outputs", "save_intermediates"], False))
    try:
        run_preflight_train(cfg.bam, cfg.gtf, cfg.ref_gtf, cfg.tmap, cfg_map)
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
        command="train",
        args_dict={
            "bam": str(cfg.bam),
            "gtf": str(cfg.gtf),
            "ref_gtf": str(cfg.ref_gtf),
            "tmap": str(cfg.tmap) if cfg.tmap else "",
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

    try:
        autosome_train_range = parse_split_policy(
            str(get_nested(cfg_map, ["stage1", "training", "split_policy"], "chr1-10"))
        )
    except ValueError as exc:
        print(f"[telos_v2] invalid split_policy: {exc}")
        return 2

    tol = int(get_nested(cfg_map, ["stage1", "training", "site_label_tolerance_bp"], 50))
    rf_cfg = get_nested(cfg_map, ["stage1", "training", "random_forest"], {}) or {}
    xgb_cfg = get_nested(cfg_map, ["stage1", "training", "xgboost"], {}) or {}
    seed = int(get_nested(cfg_map, ["stage1", "training", "random_state"], 42))
    ref_df = reference_sites_from_gtf(cfg.ref_gtf)
    df_all = pd.DataFrame(rows)
    metrics_payload: dict[str, Any] = {}

    for st in ("TSS", "TES"):
        labeled = df_all[df_all["site_type"].str.upper() == st].copy()
        if labeled.empty:
            print(f"[telos_v2] no {st} feature rows; cannot train Stage I.")
            return 2
        labeled["label"] = label_sites_by_proximity(labeled, ref_df, st, tol)
        for backend in STAGE1_BACKENDS:
            try:
                m, clf, feats = train_stage1_site_classifier(
                    labeled,
                    st,
                    autosome_train_range,
                    backend=backend,
                    rf_config=rf_cfg,
                    xgb_config=xgb_cfg,
                    random_state=seed,
                )
            except ImportError as exc:
                print(f"[telos_v2] Stage I training failed ({st}, {backend}): {exc}")
                return 2
            except ValueError as exc:
                print(f"[telos_v2] Stage I training failed ({st}, {backend}): {exc}")
                return 2
            metrics_payload[f"{st.lower()}_{backend}"] = m
            fname = stage1_bundle_path(st, backend)
            save_stage1_bundle(layout.models_dir / fname, st, backend, clf, feats)

    sites_scored = score_stage1_dataframe(df_all, layout.models_dir)
    sites_path = layout.predictions_dir / "sites.scored.tsv"
    write_sites_scored_tsv(sites_scored, sites_path)

    prob_cols = {STAGE1_BACKEND_RF: SITE_PROB_COLUMN_RF, STAGE1_BACKEND_XGB: SITE_PROB_COLUMN_XGB}
    for backend in STAGE1_BACKENDS:
        prob_col = prob_cols[backend]
        try:
            df_stage2 = build_stage2_training_frame(
                cov_path, df_all, sites_scored, cfg.tmap, site_prob_column=prob_col
            )
        except Exception as exc:
            write_train_metrics(metrics_payload, layout.reports_dir / "train_metrics.json")
            print(f"[telos_v2] Stage II feature merge failed ({backend}): {exc}")
            return 2
        if df_stage2.empty:
            write_train_metrics(metrics_payload, layout.reports_dir / "train_metrics.json")
            print(
                "[telos_v2] Stage II merged table is empty after inner-joins "
                f"(cov × TSS × TES × tmap, backend={backend}). Check transcript_id and TSS/TES coordinates."
            )
            return 2

        try:
            metrics_payload[f"stage2_{backend}"] = train_and_save_stage2(
                df_stage2,
                layout.models_dir,
                layout.predictions_dir,
                layout.reports_dir,
                autosome_train_range=autosome_train_range,
                stage1_backend_tag=backend,
                save_intermediates=save_intermediates,
            )
        except ImportError as exc:
            write_train_metrics(metrics_payload, layout.reports_dir / "train_metrics.json")
            print(f"[telos_v2] Stage II requires lightgbm: {exc}")
            return 2
        except Exception as exc:
            write_train_metrics(metrics_payload, layout.reports_dir / "train_metrics.json")
            print(f"[telos_v2] Stage II training failed ({backend}): {exc}")
            return 2

    train_metrics_path = layout.reports_dir / "train_metrics.json"
    write_train_metrics(metrics_payload, train_metrics_path)
    ranked_rf = layout.predictions_dir / transcripts_ranked_tsv_for_backend(STAGE1_BACKEND_RF)
    ranked_xgb = layout.predictions_dir / transcripts_ranked_tsv_for_backend(STAGE1_BACKEND_XGB)
    s1_list = ", ".join(
        str(layout.models_dir / stage1_bundle_path(st, b))
        for st in ("TSS", "TES")
        for b in STAGE1_BACKENDS
    )
    s2_list = ", ".join(
        str(layout.models_dir / stage2_model_joblib_for_backend(b)) for b in STAGE1_BACKENDS
    )
    summary_paths: dict[str, Path | str] = {
        "manifest": manifest_path,
        "bam": cfg.bam,
        "gtf": cfg.gtf,
        "ref_gtf": cfg.ref_gtf,
        "tmap": cfg.tmap if cfg.tmap else "",
        "outdir": cfg.outdir,
        "stage1_models": s1_list,
        "stage2_models": s2_list,
        "sites_scored": sites_path,
        "transcripts_ranked_rf": ranked_rf,
        "transcripts_ranked_xgb": ranked_xgb,
        "train_metrics_json": train_metrics_path,
    }
    summary_md = write_train_summary(
        layout.reports_dir,
        manifest_path=manifest_path,
        metrics_payload=metrics_payload,
        paths=summary_paths,
    )

    print("[telos_v2] train complete")
    print(f"  bam={cfg.bam}")
    print(f"  gtf={cfg.gtf}")
    print(f"  ref_gtf={cfg.ref_gtf}")
    print(f"  tmap={cfg.tmap}")
    print(f"  outdir={cfg.outdir}")
    print(f"  manifest={manifest_path}")
    if candidate_path is not None:
        print(f"  candidates={candidate_path}")
    if get_nested(cfg_map, ["rnaseqtools", "run_get_cov"], True):
        print(f"  cov_tsv={cov_path}")
    if tss_path is not None and tes_path is not None:
        print(f"  stage1_tss={tss_path}")
        print(f"  stage1_tes={tes_path}")
    print(f"  stage1_models={s1_list}")
    print(f"  train_metrics={train_metrics_path}")
    print(f"  summary_md={summary_md}")
    print(f"  sites_scored={sites_path}")
    print(f"  stage2_models={s2_list}")
    print(f"  transcripts_ranked_rf={ranked_rf}")
    print(f"  transcripts_ranked_xgb={ranked_xgb}")
    return 0
