"""
Stacked PDF figures for feature-importance facets and feature-window sweeps.

Aligned with ``grouped_aupr_bars`` layout: large fonts, no titles, shared legend,
reduced x-axis redundancy, PDF output under ``plot_v2/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from telos_v2.analysis.stage1_importance import STAGE1_FEATURE_GROUP_ORDER
from telos_v2.plotting.grouped_aupr_bars import (
    COLOR_BASELINE,
    COLOR_TELOS_RF,
    COLOR_TELOS_XGB,
    DATA_TYPE_PLOT_ORDER,
    DATA_TYPE_TITLE,
    MODEL_BASELINE_LABEL,
    MODEL_RF_LABEL,
    MODEL_XGB_LABEL,
    TRANSCRIPT_PR_AUC_DIVISOR,
    _add_metric_row_label_fig,
    _figure_legend,
    apply_plot_style,
)

_BAR_DPI = 300
_FONT_TICK = 14
_FONT_FEATURE_Y = 11
_FONT_LEGEND = 15
_FONT_XLABEL = 14
_FONT_MODALITY = 15
_FONT_SECTION = 16
_FONT_SIDE = 14
# 8×1 stacked panels: TSS (PacBio→SR) then TES (PacBio→SR).
_FI_FIG_W = 12.0
_FI_PANEL_H = 2.2
_FI_BAR_STEP = 1.45
_FI_BAR_HEIGHT = 0.52
_FI_SIDE_LABEL_X = 0.035
_FI_SECTION_HEADER_X = 0.52
_FI_SECTION_HEADER_PAD = 0.01
_STACKED_HSPACE = 0.22
_LEGEND_FIG_Y = 0.985
_FI_TOP = 0.93
_FI_SITE_TYPES = ("TSS", "TES")

FI_GROUP_COLORS = {
    "read_alignment": "#4C78A8",
    "soft_clip": "#72B7B2",
    "nucleotide_composition": "#B279A2",
    "k3_statistics": "#F58518",
    "kmer_counts": "#FF9DA6",
    "other": "#9D9D9D",
}

FI_GROUP_LABELS = {
    "read_alignment": "read alignment",
    "soft_clip": "soft-clip",
    "nucleotide_composition": "nucleotide composition",
    "k3_statistics": "3-mer statistics",
    "kmer_counts": "k-mer counts",
    "other": "other",
}

_FI_HEATMAP_W = 10.0
_FI_HEATMAP_CBAR_RATIO = 0.05
_FI_HEATMAP_ROW_SCALE = 0.28
_FI_HEATMAP_MIN_H = 8.0
_FI_HEATMAP_LEFT = 0.28
_FI_HEATMAP_RIGHT = 0.88
_FI_HEATMAP_TOP = 0.82
_FI_HEATMAP_BOTTOM = 0.08
_FI_HEATMAP_HSPACE = 0.34
_FI_SECTION_TOP_PAD = 0.022
_FI_HEATMAP_FEATURE_FONT = 10
_FI_HEATMAP_MODALITY_FONT = 14
_STACKED_TOP = 0.90

WINDOW_METRIC_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    ("stage1_tss", "TSS", "stage1_test_aupr_tss_rf", "stage1_test_aupr_tss_xgb", "stage1_test_aupr_tss_baseline"),
    ("stage1_tes", "TES", "stage1_test_aupr_tes_rf", "stage1_test_aupr_tes_xgb", "stage1_test_aupr_tes_baseline"),
    ("transcript_pr", "Transcript", "transcript_pr_auc_model_rf", "transcript_pr_auc_model_xgb", "transcript_pr_auc_baseline"),
)

WINDOW_LINE_STYLES: tuple[tuple[str, str, str, str], ...] = (
    (MODEL_BASELINE_LABEL, "baseline", COLOR_BASELINE, "^"),
    (MODEL_RF_LABEL, "rf", COLOR_TELOS_RF, "o"),
    (MODEL_XGB_LABEL, "xgb", COLOR_TELOS_XGB, "s"),
)


def _pdf_path(path: Path) -> Path:
    return path.with_suffix(".pdf")


def _load_fi_long(fi_dir: Path) -> pd.DataFrame:
    long_path = fi_dir / "stage1_builtin_importance_long.tsv"
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing {long_path}; run stage1_feature_importance_gencode.py first.")
    return pd.read_csv(long_path, sep="\t")


def _fi_panel_order() -> list[tuple[str, str]]:
    """TSS × modalities, then TES × modalities."""
    panels: list[tuple[str, str]] = []
    for st in _FI_SITE_TYPES:
        for dt in DATA_TYPE_PLOT_ORDER:
            panels.append((st, dt))
    return panels


def _pretty_feature_name(name: str) -> str:
    """Human-readable feature labels (no truncation or mid-word breaks)."""
    s = str(name)
    for old, new in (
        ("nucleotide_composition", "Nucleotide comp"),
        ("read_alignment", "Alignment"),
        ("soft_clip", "Soft clip"),
        ("k3_statistics", "3-mer stats"),
        ("kmer_counts", "K-mer counts"),
        ("upstream", "↑"),
        ("downstream", "↓"),
        ("fraction", "Frac"),
        ("gradient", "Grad"),
        ("sharpness", "Sharp"),
        ("coverage", "Cov"),
        ("_", " "),
    ):
        s = s.replace(old, new)
    return " ".join(s.split())


def _fi_heatmap_cmap() -> mcolors.LinearSegmentedColormap:
    """Warm sequential scale (low = pale, high = deep) — suited to non-negative importance."""
    colors = ["#ffffff", "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]
    cmap = mcolors.LinearSegmentedColormap.from_list("telos_fi", colors, N=256)
    cmap.set_bad(color="#ececec")
    return cmap


def _fi_axis(axes: np.ndarray, idx: int) -> plt.Axes:
    if axes.ndim == 2:
        return axes[idx, 0]
    return axes[idx]


def _add_side_label(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    pos = ax.get_position()
    fig.text(
        _FI_SIDE_LABEL_X,
        pos.y0 + pos.height / 2,
        text,
        transform=fig.transFigure,
        rotation=90,
        ha="center",
        va="center",
        fontsize=_FONT_SIDE,
        fontweight="bold",
    )


def _add_section_header(fig: plt.Figure, axes: np.ndarray, row_idx: int, text: str) -> None:
    ax = _fi_axis(axes, row_idx)
    pos = ax.get_position()
    fig.text(
        _FI_SECTION_HEADER_X,
        pos.y1 + _FI_SECTION_HEADER_PAD,
        text,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=_FONT_SECTION,
        fontweight="bold",
    )


def _fi_global_vmax(long_df: pd.DataFrame, *, backend: str) -> float:
    sub = long_df[long_df["backend"].str.lower() == backend.lower()]
    vals = pd.to_numeric(sub["importance_norm"], errors="coerce").dropna()
    if vals.empty:
        return 1.0
    q = float(vals.quantile(0.99))
    if not np.isfinite(q) or q <= 0:
        q = float(vals.max())
    return max(q, 0.01)


def _fi_union_feature_rows(
    long_df: pd.DataFrame,
    *,
    backend: str,
    site_type: str,
    top_n: int,
    avg_windows: bool,
) -> tuple[list[str], dict[str, str]]:
    """Union of top-N features per modality; ordered by mean importance (high → low)."""
    pieces: list[pd.DataFrame] = []
    for dt in DATA_TYPE_PLOT_ORDER:
        cell = _fi_agg(
            long_df,
            backend=backend,
            site_type=site_type,
            data_type=dt,
            avg_windows=avg_windows,
        )
        if cell.empty:
            continue
        top = cell.nlargest(top_n, "importance_norm")
        pieces.append(top[["feature", "group", "importance_norm"]].assign(data_type=dt))

    if not pieces:
        return [], {}

    combined = pd.concat(pieces, ignore_index=True)
    group_map = (
        combined.drop_duplicates("feature")
        .set_index("feature")["group"]
        .astype(str)
        .to_dict()
    )
    rank = (
        combined.groupby("feature", as_index=False)["importance_norm"]
        .mean()
        .sort_values("importance_norm", ascending=False)
    )
    return rank["feature"].tolist(), group_map


def _fi_heatmap_matrix(
    long_df: pd.DataFrame,
    *,
    backend: str,
    site_type: str,
    top_n: int,
    avg_windows: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Feature × modality matrix (NaN where feature not in that modality's top-N)."""
    features, group_map = _fi_union_feature_rows(
        long_df,
        backend=backend,
        site_type=site_type,
        top_n=top_n,
        avg_windows=avg_windows,
    )
    if not features:
        return pd.DataFrame(), features, group_map

    cols = list(DATA_TYPE_PLOT_ORDER)
    mat = pd.DataFrame(np.nan, index=features, columns=cols, dtype=float)
    for dt in cols:
        cell = _fi_agg(
            long_df,
            backend=backend,
            site_type=site_type,
            data_type=dt,
            avg_windows=avg_windows,
        )
        if cell.empty:
            continue
        sub = cell.set_index("feature")["importance_norm"]
        for feat in features:
            if feat in sub.index:
                mat.at[feat, dt] = float(sub.loc[feat])
    return mat, features, group_map


def _add_heatmap_section_header(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 + pos.width * 0.5,
        pos.y1 + _FI_SECTION_TOP_PAD,
        text,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=_FONT_SECTION,
        fontweight="bold",
    )


def _draw_heatmap_panel(
    ax: plt.Axes,
    mat: pd.DataFrame,
    *,
    vmax: float,
    group_map: dict[str, str],
) -> plt.cm.ScalarMappable:
    data = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = _fi_heatmap_cmap()
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )
    n_feat = len(mat.index)
    if n_feat:
        ax.set_yticks(np.arange(n_feat))
        ylabels = ax.set_yticklabels(
            [_pretty_feature_name(f) for f in mat.index],
            fontsize=_FI_HEATMAP_FEATURE_FONT,
            ha="right",
        )
        for lbl, feat in zip(ylabels, mat.index):
            grp = group_map.get(feat, "other")
            lbl.set_color(FI_GROUP_COLORS.get(grp, FI_GROUP_COLORS["other"]))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(
        [DATA_TYPE_TITLE.get(c, c) for c in mat.columns],
        fontsize=_FI_HEATMAP_MODALITY_FONT,
        fontweight="bold",
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.tick_params(axis="y", length=0)
    return im


def plot_feature_importance_heatmap(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    backend: str,
    top_n: int = 10,
    avg_windows: bool = False,
) -> None:
    """One PDF per model: TSS and TES heatmaps (features × modalities)."""
    apply_plot_style()
    vmax = _fi_global_vmax(long_df, backend=backend)
    matrices: dict[str, pd.DataFrame] = {}
    groups: dict[str, dict[str, str]] = {}
    for st in _FI_SITE_TYPES:
        mat, _feats, gmap = _fi_heatmap_matrix(
            long_df,
            backend=backend,
            site_type=st,
            top_n=top_n,
            avg_windows=avg_windows,
        )
        matrices[st] = mat
        groups[st] = gmap

    n_feat_max = max((len(m.index) for m in matrices.values()), default=1)
    fig_h = max(
        _FI_HEATMAP_MIN_H,
        _FI_HEATMAP_ROW_SCALE * n_feat_max * len(_FI_SITE_TYPES),
    )
    fig = plt.figure(figsize=(_FI_HEATMAP_W, fig_h), dpi=_BAR_DPI)
    gs = GridSpec(
        len(_FI_SITE_TYPES),
        2,
        figure=fig,
        width_ratios=[1.0, _FI_HEATMAP_CBAR_RATIO],
        hspace=_FI_HEATMAP_HSPACE,
        wspace=0.02,
        left=_FI_HEATMAP_LEFT,
        right=_FI_HEATMAP_RIGHT,
        top=_FI_HEATMAP_TOP,
        bottom=_FI_HEATMAP_BOTTOM,
    )

    mappable = None
    for i, st in enumerate(_FI_SITE_TYPES):
        mat = matrices[st]
        if mat.empty:
            continue
        ax_hm = fig.add_subplot(gs[i, 0])
        mappable = _draw_heatmap_panel(ax_hm, mat, vmax=vmax, group_map=groups[st])
        _add_heatmap_section_header(fig, ax_hm, st)

    if mappable is not None:
        cax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Mean normalized importance", fontsize=_FONT_XLABEL)
        cbar.ax.tick_params(labelsize=_FONT_TICK)

    handles = [
        mpatches.Patch(color=FI_GROUP_COLORS[g], label=FI_GROUP_LABELS.get(g, g))
        for g in STAGE1_FEATURE_GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=_FONT_LEGEND,
        frameon=False,
    )

    out_pdf = _pdf_path(out_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

    # Long table for caption / supplement
    rows: list[pd.DataFrame] = []
    for st in _FI_SITE_TYPES:
        mat = matrices[st]
        if mat.empty:
            continue
        long = mat.reset_index(names="feature").melt(
            id_vars="feature",
            var_name="data_type",
            value_name="importance_norm",
        )
        long["site_type"] = st
        long["backend"] = backend.lower()
        rows.append(long)
    if rows:
        tbl = pd.concat(rows, ignore_index=True)
        tbl.to_csv(out_pdf.with_suffix(".csv"), index=False, sep="\t")


def _fi_agg(
    long_df: pd.DataFrame,
    *,
    backend: str,
    site_type: str,
    data_type: str,
    avg_windows: bool,
) -> pd.DataFrame:
    sub = long_df[
        (long_df["backend"].str.lower() == backend.lower())
        & (long_df["site_type"].str.upper() == site_type.upper())
        & (long_df["data_type"].astype(str).str.lower() == data_type.lower())
    ].copy()
    if avg_windows and "window_size" in sub.columns:
        return (
            sub.groupby(["feature", "group"], as_index=False)["importance_norm"]
            .mean()
            .sort_values("importance_norm", ascending=False)
        )
    return (
        sub.groupby(["feature", "group"], as_index=False)["importance_norm"]
        .mean()
        .sort_values("importance_norm", ascending=False)
    )

def plot_feature_importance_stacked(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    backend: str,
    top_n: int = 10,
    avg_windows: bool = False,
) -> None:
    """
    One PDF per model: 8×1 stacked panels — TSS (PacBio→SR) then TES (PacBio→SR).
    """
    apply_plot_style()
    panels = _fi_panel_order()
    nrows = len(panels)
    x_max = _fi_global_vmax(long_df, backend=backend) * 1.05

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(_FI_FIG_W, _FI_PANEL_H * nrows),
        dpi=_BAR_DPI,
        sharex=True,
        squeeze=False,
    )

    for idx, (st, dt) in enumerate(panels):
        ax = _fi_axis(axes, idx)
        cell = _fi_agg(
            long_df, backend=backend, site_type=st, data_type=dt, avg_windows=avg_windows
        )
        cell = cell.nlargest(top_n, "importance_norm").sort_values("importance_norm", ascending=True)
        if cell.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(cell)) * _FI_BAR_STEP
        colors = [FI_GROUP_COLORS.get(g, FI_GROUP_COLORS["other"]) for g in cell["group"]]
        ax.barh(y, cell["importance_norm"], color=colors, height=_FI_BAR_HEIGHT)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_pretty_feature_name(f) for f in cell["feature"]],
            fontsize=_FONT_FEATURE_Y,
            ha="right",
        )
        ax.tick_params(axis="y", labelsize=_FONT_FEATURE_Y, pad=4, length=0)
        ax.tick_params(axis="x", labelsize=_FONT_TICK)
        ymax = y[-1] + _FI_BAR_STEP * 0.55 if len(y) else _FI_BAR_STEP
        ax.set_ylim(-_FI_BAR_STEP * 0.35, ymax)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        if idx < nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)

    _fi_axis(axes, nrows - 1).set_xlabel("Mean normalized importance", fontsize=_FONT_XLABEL)

    handles = [
        mpatches.Patch(color=FI_GROUP_COLORS[g], label=FI_GROUP_LABELS.get(g, g))
        for g in STAGE1_FEATURE_GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, _LEGEND_FIG_Y),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=_FONT_LEGEND,
        frameon=False,
    )
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.06, top=_FI_TOP, hspace=0.15)

    _add_section_header(fig, axes, 0, "TSS")
    _add_section_header(fig, axes, len(DATA_TYPE_PLOT_ORDER), "TES")
    for idx, (_st, dt) in enumerate(panels):
        ax = _fi_axis(axes, idx)
        if ax.get_visible():
            _add_side_label(fig, ax, DATA_TYPE_TITLE.get(dt, dt))

    out_pdf = _pdf_path(out_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _load_window_summaries(root: Path) -> pd.DataFrame:
    import re

    pat = re.compile(
        r"^(?P<data_type>.+?)__ws(?P<window_size>\d+)__train_gencode__test_gencode$"
    )
    rows: list[pd.DataFrame] = []
    for summary in sorted(root.glob("*/reports/benchmark_summary.csv")):
        run_id = summary.parent.parent.name
        m = pat.match(run_id.strip())
        if not m:
            continue
        df = pd.read_csv(summary)
        if df.empty:
            continue
        df["data_type"] = m.group("data_type").strip()
        df["window_size"] = int(m.group("window_size"))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No window-sweep summaries under {root}")
    out = pd.concat(rows, axis=0, ignore_index=True)
    # transcript PR: gtfcuff AUC is in percent² units; normalize to 0–1 AuPR.
    if "transcript_pr_auc_model_rf" in out.columns:
        for col in ("transcript_pr_auc_model_rf", "transcript_pr_auc_model_xgb", "transcript_pr_auc_baseline"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") / TRANSCRIPT_PR_AUC_DIVISOR
    return out


def _mean_by_window(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    s = sub[["window_size", col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    return (
        s.dropna(subset=[col])
        .groupby("window_size", as_index=False)[col]
        .mean()
        .sort_values("window_size")
    )


def plot_window_sweep_stacked(df: pd.DataFrame, out_path: Path) -> None:
    """One PDF: metrics stacked vertically; each row = 4 data-type line panels."""
    apply_plot_style()
    n_metrics = len(WINDOW_METRIC_ROWS)
    n_dt = len(DATA_TYPE_PLOT_ORDER)
    fig, axes = plt.subplots(
        n_metrics,
        n_dt,
        figsize=(3.2 * n_dt, 3.0 * n_metrics),
        dpi=_BAR_DPI,
        sharex="col",
    )
    if n_metrics == 1:
        axes = np.array([[axes]]) if n_dt == 1 else np.array([axes])
    if n_dt == 1:
        axes = axes.reshape(n_metrics, 1)

    for i, (stem, row_lbl, rf_c, xgb_c, bl_c) in enumerate(WINDOW_METRIC_ROWS):
        for j, dt in enumerate(DATA_TYPE_PLOT_ORDER):
            ax = axes[i, j]
            sub = df[df["data_type"].astype(str).str.lower() == dt]
            for _lbl, key, color, marker in WINDOW_LINE_STYLES:
                col = {"rf": rf_c, "xgb": xgb_c, "baseline": bl_c}[key]
                if col not in sub.columns:
                    continue
                agg = _mean_by_window(sub, col)
                if agg.empty:
                    continue
                ax.plot(
                    agg["window_size"],
                    agg[col],
                    marker=marker,
                    color=color,
                    linewidth=2,
                    markersize=7,
                )
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.tick_params(labelsize=_FONT_TICK)
            if i < n_metrics - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("window (bp)", fontsize=_FONT_TICK)
            if i == 0:
                ax.text(
                    0.5,
                    1.03,
                    DATA_TYPE_TITLE.get(dt, dt),
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=_FONT_MODALITY,
                    fontweight="bold",
                )

    _figure_legend(fig)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.08, top=_STACKED_TOP, hspace=0.35, wspace=0.28)
    for i, (_stem, row_lbl, *_cols) in enumerate(WINDOW_METRIC_ROWS):
        ax0 = axes[i, 0]
        if ax0.get_visible():
            _add_metric_row_label_fig(fig, ax0, row_lbl)
    out_pdf = _pdf_path(out_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

    # table
    tbl_cols = [s[2] for s in WINDOW_METRIC_ROWS] + [s[3] for s in WINDOW_METRIC_ROWS]
    present = [c for c in tbl_cols if c in df.columns]
    if present:
        tbl = (
            df.groupby(["data_type", "window_size"], as_index=False)[present]
            .mean(numeric_only=True)
            .sort_values(["data_type", "window_size"])
        )
        tbl.to_csv(out_pdf.parent / "mean_metrics_by_window_size.csv", index=False)


def run_feature_importance_plots(
    fi_dir: Path,
    outdir: Path,
    *,
    top_n: int = 10,
    avg_windows: bool = False,
    include_stacked: bool = True,
) -> None:
    long_df = _load_fi_long(fi_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for backend in ("rf", "xgb"):
        plot_feature_importance_heatmap(
            long_df,
            outdir / f"stage1_fi_top{top_n}_heatmap_{backend}.pdf",
            backend=backend,
            top_n=top_n,
            avg_windows=avg_windows,
        )
        if include_stacked:
            plot_feature_importance_stacked(
                long_df,
                outdir / f"stage1_fi_top{top_n}_stacked_{backend}.pdf",
                backend=backend,
                top_n=top_n,
                avg_windows=avg_windows,
            )


def run_window_plots(root: Path, outdir: Path) -> None:
    df = _load_window_summaries(root)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_window_sweep_stacked(df, outdir / "window_sweep_stacked.pdf")
