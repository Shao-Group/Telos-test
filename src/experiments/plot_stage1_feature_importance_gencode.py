"""
Plot top Stage I feature importances (facet grid per model).

Reads ``stage1_builtin_importance_long.tsv`` from ``stage1_feature_importance_gencode.py`` and writes
one facet figure per backend (RF, XGB): data_type × site_type panels with top-N features.

Usage::

    PYTHONPATH=src python src/experiments/plot_stage1_feature_importance_gencode.py

    PYTHONPATH=src python src/experiments/plot_stage1_feature_importance_gencode.py \\
      --run-root runs/stage1_feature_importance_gencode --backend rf --top-n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from telos_repro.analysis.stage1_importance import STAGE1_FEATURE_GROUP_ORDER

DATA_TYPE_LABELS = {
    "sr": "Short-read",
    "cdna": "ONT cDNA",
    "drna": "ONT dRNA",
    "pacbio": "PacBio",
}

GROUP_ORDER = list(STAGE1_FEATURE_GROUP_ORDER)

GROUP_LABELS = {
    "read_alignment": "read alignment",
    "soft_clip": "soft-clip statistic",
    "nucleotide_composition": "nucleotide composition",
    "k3_statistics": "3-mer statistics",
    "kmer_counts": "k-mer counts",
    "other": "other",
}

GROUP_COLORS = {
    "read_alignment": "#4C78A8",
    "soft_clip": "#72B7B2",
    "nucleotide_composition": "#B279A2",
    "k3_statistics": "#F58518",
    "kmer_counts": "#FF9DA6",
    "other": "#9D9D9D",
}


def _group_label(name: str) -> str:
    return GROUP_LABELS.get(name, name.replace("_", " "))


def _load_long_table(fi_dir: Path) -> pd.DataFrame:
    fi_dir = fi_dir.resolve()
    long_path = fi_dir / "stage1_builtin_importance_long.tsv"
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing {long_path}; run stage1_feature_importance_gencode.py first.")
    long_df = pd.read_csv(long_path, sep="\t")
    if "window_size" in long_df.columns:
        long_df["window_size"] = pd.to_numeric(long_df["window_size"], errors="coerce")
    return long_df


def _agg_long(
    long_df: pd.DataFrame,
    *,
    backend: str,
    site_type: str,
    avg_windows: bool,
) -> pd.DataFrame:
    sub = long_df[
        (long_df["backend"].str.lower() == backend.lower())
        & (long_df["site_type"].str.upper() == site_type.upper())
    ].copy()
    keys = ["data_type", "feature", "group"]
    if avg_windows:
        return (
            sub.groupby(keys, as_index=False)["importance_norm"]
            .mean()
            .sort_values("importance_norm", ascending=False)
        )
    keys = ["data_type", "window_size", "feature", "group"]
    return sub.groupby(keys, as_index=False)["importance_norm"].mean()


def plot_top_features_facets(
    long_df: pd.DataFrame,
    *,
    out_path: Path,
    backend: str,
    top_n: int,
    avg_windows: bool,
) -> None:
    """Delegate to stacked PDF plotter in ``telos_repro.plotting.feature_window_plots``."""
    from telos_repro.plotting.feature_window_plots import plot_feature_importance_stacked

    plot_feature_importance_stacked(
        long_df,
        out_path,
        backend=backend,
        top_n=top_n,
        avg_windows=avg_windows,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Plot top Stage I feature importances (facet grids).")
    p.add_argument("--run-root", type=Path, default=Path("runs/stage1_feature_importance_gencode"))
    p.add_argument("--fi-dir", type=Path, default=None, help="Feature importance TSV directory.")
    p.add_argument("--plot-dir", type=Path, default=None)
    p.add_argument("--backend", choices=("rf", "xgb", "both"), default="both")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--avg-windows",
        action="store_true",
        help="Average over window_size when multiple values exist.",
    )
    args = p.parse_args()
    run_root = args.run_root.resolve()
    fi_dir = (args.fi_dir or (run_root / "reports" / "feature_importance")).resolve()
    plot_dir = (args.plot_dir or (run_root / "plots" / "feature_importance")).resolve()
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        long_df = _load_long_table(fi_dir)
        avg_windows = args.avg_windows
        backends = ("rf", "xgb") if args.backend == "both" else (args.backend,)
        from telos_repro.plotting.feature_window_plots import (
            plot_feature_importance_heatmap,
            plot_feature_importance_stacked,
        )

        for be in backends:
            plot_feature_importance_heatmap(
                long_df,
                plot_dir / f"stage1_fi_top{args.top_n}_heatmap_{be}.pdf",
                backend=be,
                top_n=args.top_n,
                avg_windows=avg_windows,
            )
            plot_feature_importance_stacked(
                long_df,
                plot_dir / f"stage1_fi_top{args.top_n}_stacked_{be}.pdf",
                backend=be,
                top_n=args.top_n,
                avg_windows=avg_windows,
            )
        print(f"[plot] figures written to {plot_dir}")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
