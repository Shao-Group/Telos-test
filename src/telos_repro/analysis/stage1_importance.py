"""
Stage I feature importance: built-in tree importances and optional permutation on validation sites.

Designed for benchmark matrices (data type × window size × GENCODE train) where models live under
``<cell>/train/models/stage1_{tss,tes}_{rf,xgb}_model.joblib``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from telos.models.stage1_train import (
    _final_estimator,
    build_stage1_feature_matrix,
    load_stage1_bundle,
)
from telos.models import STAGE1_BACKEND_RF, STAGE1_BACKEND_XGB

# Semantic groups for Stage I columns (paper Table: features for TSS/TES classification).
STAGE1_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "read_alignment": (
        "total_reads",
        "read_start_density",
        "read_end_density",
        "mean_mapq",
        "std_mapq",
        "strand_ratio",
        "coverage_before",
        "coverage_after",
        "delta_coverage",
        "coverage_gradient_sharpness",
        "max_coverage_gradient",
        "local_avg_coverage",
        "coverage_efficiency",
        "coverage_log_ratio",
        "coverage_ratio",
        "five_prime_degradation_score",
        "normalized_delta_coverage",
        "norm_read_density",
        "nearest_splice_dist",
        "read_start_clustering",
        "read_start_coefficient_variation",
        "read_start_variance",
        "up_down_stream_ratio",
        "upstream_downstream_ratio",
        "weighted_read_start_density",
    ),
    "soft_clip": (
        "start_soft_clip_mean",
        "end_soft_clip_mean",
        "start_soft_clip_max",
        "end_soft_clip_max",
        "start_soft_clip_median",
        "end_soft_clip_median",
        "start_soft_clip_count",
        "end_soft_clip_count",
        "softclip_bias",
        "softclip_sparsity",
        "softclip_length_variance",
        "softclip_length_skewness",
        "avg_clip_length",
        "max_clip_length",
        "num_clips",
        "start_entropy",
        "end_entropy",
    ),
    "nucleotide_composition": (
        "gc_content",
        "purine_ratio",
        "max_polyA",
        "max_polyC",
        "max_polyG",
        "max_polyT",
    ),
}

STAGE1_FEATURE_GROUP_ORDER: tuple[str, ...] = (
    "read_alignment",
    "soft_clip",
    "nucleotide_composition",
    "k3_statistics",
    "kmer_counts",
    "other",
)

DEFAULT_STAGE1_WINDOW_SIZE = 100
FI_GENCODE_SHARED_TRAIN_SUBDIR = "_shared_train"


def stage1_feature_group(feature: str) -> str:
    """Map a feature column name to a coarse semantic group."""
    for group, names in STAGE1_FEATURE_GROUPS.items():
        if feature in names:
            return group
    if feature.startswith("k3_"):
        return "k3_statistics"
    if feature.startswith("kmer_"):
        return "kmer_counts"
    return "other"


def builtin_importances_from_bundle(bundle_path: Path) -> pd.DataFrame:
    """
    Extract ``feature_importances_`` from a saved Stage I joblib bundle (RF or XGB pipeline).

    Returns columns ``feature``, ``importance`` (non-negative, sums to 1 for RF).
    """
    bundle = load_stage1_bundle(bundle_path)
    feats: list[str] = list(bundle["feature_names"])
    est = _final_estimator(bundle["model"])
    if not hasattr(est, "feature_importances_"):
        raise TypeError(f"Model in {bundle_path} has no feature_importances_: {type(est)}")
    imp = np.asarray(est.feature_importances_, dtype=float)
    if len(imp) != len(feats):
        raise ValueError(f"Importance length {len(imp)} != feature_names {len(feats)} in {bundle_path}")
    df = pd.DataFrame({"feature": feats, "importance": imp})
    df["group"] = df["feature"].map(stage1_feature_group)
    total = float(df["importance"].sum())
    if total > 0:
        df["importance_norm"] = df["importance"] / total
    else:
        df["importance_norm"] = 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def group_importance_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized importances by :func:`stage1_feature_group`."""
    if "importance_norm" not in importance_df.columns:
        imp = importance_df.copy()
        t = float(imp["importance"].sum()) or 1.0
        imp["importance_norm"] = imp["importance"] / t
    else:
        imp = importance_df
    g = (
        imp.groupby("group", as_index=False)["importance_norm"]
        .sum()
        .rename(columns={"importance_norm": "group_importance"})
        .sort_values("group_importance", ascending=False)
    )
    return g


def permutation_importance_stage1(
    model: Any,
    feature_names: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int | None = 8000,
) -> pd.DataFrame:
    """
    Validation-chromosome permutation importance (macro AUPR drop when each column is shuffled).

    Subsamples rows when ``len(X_val) > max_samples`` for speed. Requires both classes in ``y_val``.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, make_scorer

    y = y_val.astype(int)
    if y.nunique() < 2:
        raise ValueError("permutation_importance_stage1 requires both classes in y_val")

    X = build_stage1_feature_matrix(X_val, feature_names)
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    scorer = make_scorer(average_precision_score, response_method="predict_proba")

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scorer,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df["group"] = df["feature"].map(stage1_feature_group)
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def parse_gencode_benchmark_cell_dir(name: str) -> dict[str, str] | None:
    """
    Parse ``{data_type}__ws{N}__train_gencode__test_gencode`` directory names from window sweeps.

    Returns None if the pattern does not match.
    """
    m = re.fullmatch(r"(?P<data_type>[a-z0-9]+)__ws(?P<window_size>\d+)__train_gencode__test_gencode", name)
    if not m:
        return None
    return {
        "data_type": m.group("data_type"),
        "window_size": m.group("window_size"),
        "train_annotation": "gencode",
        "test_annotation": "gencode",
    }


def parse_cross_annotation_gencode_cell_dir(name: str) -> dict[str, str] | None:
    """
    Parse ``{data_type}__train_gencode__test_gencode`` cross-annotation matrix cells.

    Window size is fixed in config (see :data:`DEFAULT_STAGE1_WINDOW_SIZE`), not in the dir name.
    """
    m = re.fullmatch(
        r"(?P<data_type>[a-z0-9]+)__train_gencode__test_gencode",
        name,
    )
    if not m:
        return None
    return {
        "data_type": m.group("data_type"),
        "window_size": str(DEFAULT_STAGE1_WINDOW_SIZE),
        "train_annotation": "gencode",
        "test_annotation": "gencode",
    }


def shared_train_models_dir(
    run_root: Path,
    data_type: str,
    *,
    train_annotation: str = "gencode",
    shared_train_subdir: str = FI_GENCODE_SHARED_TRAIN_SUBDIR,
) -> Path:
    """``<run_root>/<shared_train_subdir>/<data_type>__train_<anno>/models``."""
    return run_root / shared_train_subdir / f"{data_type}__train_{train_annotation}" / "models"


def cross_annotation_shared_models_dir(run_root: Path, data_type: str) -> Path:
    """Legacy alias for cross-annotation repro layout."""
    return shared_train_models_dir(
        run_root,
        data_type,
        shared_train_subdir="_cross_annotation_shared_train",
    )


def iter_stage1_model_paths(models_dir: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield ``(site_type, backend, path)`` for each ``stage1_*_model.joblib`` under ``models_dir``."""
    for site in ("tss", "tes"):
        for backend in (STAGE1_BACKEND_RF, STAGE1_BACKEND_XGB):
            p = models_dir / f"stage1_{site}_{backend}_model.joblib"
            if p.is_file():
                yield site.upper(), backend, p


def collect_builtin_importance_from_run_root(
    run_root: Path,
    *,
    models_subdir: Path | None = None,
    shared_train: bool = False,
    shared_train_subdir: str = FI_GENCODE_SHARED_TRAIN_SUBDIR,
    cross_annotation: bool | None = None,
    gencode_only: bool = True,
) -> pd.DataFrame:
    """
    Walk ``run_root`` and emit one row per (cell, site_type, backend, feature).

    With ``shared_train=True``, reads models from
    ``<shared_train_subdir>/<data_type>__train_gencode/models`` (one row per data type).
    Otherwise supports window-sweep cells ``*__ws{N}__train_gencode__test_gencode`` with
    ``train/models`` under each cell.
    """
    if cross_annotation is not None:
        shared_train = cross_annotation
        if cross_annotation:
            shared_train_subdir = "_cross_annotation_shared_train"
    models_subdir = models_subdir or Path("train/models")
    roots: list[tuple[dict[str, str], Path]] = []

    if shared_train:
        shared = run_root / shared_train_subdir
        if shared.is_dir():
            for child in sorted(shared.iterdir()):
                if not child.is_dir():
                    continue
                m = re.fullmatch(r"(?P<data_type>[a-z0-9]+)__train_gencode", child.name)
                if m is None:
                    continue
                md = child / "models"
                if not md.is_dir():
                    continue
                meta = {
                    "data_type": m.group("data_type"),
                    "window_size": str(DEFAULT_STAGE1_WINDOW_SIZE),
                    "train_annotation": "gencode",
                    "test_annotation": "gencode",
                }
                roots.append((meta, md))
    else:
        meta = parse_gencode_benchmark_cell_dir(run_root.name)
        if meta is not None and (run_root / models_subdir).is_dir():
            roots.append((meta, run_root / models_subdir))
        else:
            for child in sorted(run_root.iterdir()):
                if not child.is_dir():
                    continue
                m = parse_gencode_benchmark_cell_dir(child.name)
                if m is None:
                    continue
                if gencode_only and (
                    m.get("train_annotation") != "gencode" or m.get("test_annotation") != "gencode"
                ):
                    continue
                md = child / models_subdir
                if md.is_dir():
                    roots.append((m, md))

    rows: list[pd.DataFrame] = []
    for meta, models_dir in roots:
        for site_type, backend, bundle_path in iter_stage1_model_paths(models_dir):
            fi = builtin_importances_from_bundle(bundle_path)
            fi["site_type"] = site_type
            fi["backend"] = backend
            for k, v in meta.items():
                fi[k] = v
            fi["model_path"] = str(bundle_path)
            rows.append(fi)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def iter_cross_annotation_gencode_cells(run_root: Path) -> list[Path]:
    """Matrix cell dirs ``<data_type>__train_gencode__test_gencode`` under ``run_root``."""
    cells: list[Path] = []
    for child in sorted(run_root.iterdir()):
        if child.is_dir() and parse_cross_annotation_gencode_cell_dir(child.name) is not None:
            cells.append(child)
    return cells
