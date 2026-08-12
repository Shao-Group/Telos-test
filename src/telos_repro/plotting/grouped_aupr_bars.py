"""
Grouped AUPR bar plots (baseline vs Telos-RF vs Telos-XGB) for v2 benchmark summaries.

Used by ``telos_repro.plotting.plot_experiments`` (outputs under ``plot_v2/``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from telos_repro.plotting.v1_style_labels import (
    CROSS_ANNOTATION_TOOL_PANELS,
    PANEL_DISPLAY,
    V1_GROUP_ORDER,
    V1_GROUP_TITLE,
    annotate_v1_labels,
    expected_panel_size,
    row_matches_tool_panel,
)

# gtfcuff transcript PR AUC integrates recall(%)×precision(%); v1 normalizes by 10000 to 0–1 AuPR.
TRANSCRIPT_PR_AUC_DIVISOR = 10000.0

# (stem, title, rf_col, xgb_col, baseline_col, scale_transcript_pr)
BenchmarkMetricSpec = tuple[str, str, str, str, str, bool]

CORE_BENCHMARK_METRICS: tuple[BenchmarkMetricSpec, ...] = (
    (
        "stage1_tss",
        "Stage I TSS (test AUPR)",
        "stage1_test_aupr_tss_rf",
        "stage1_test_aupr_tss_xgb",
        "stage1_test_aupr_tss_baseline",
        False,
    ),
    (
        "stage1_tes",
        "Stage I TES (test AUPR)",
        "stage1_test_aupr_tes_rf",
        "stage1_test_aupr_tes_xgb",
        "stage1_test_aupr_tes_baseline",
        False,
    ),
    (
        "transcript_pr",
        "Transcript PR (AUC vs ref)",
        "transcript_pr_auc_model_rf",
        "transcript_pr_auc_model_xgb",
        "transcript_pr_auc_baseline",
        True,
    ),
)

# (stem, title, filter kwargs, scale_transcript_pr)
NovelMetricSpec = tuple[str, str, dict[str, Any], bool]

NOVEL_METRIC_SPECS: tuple[NovelMetricSpec, ...] = (
    (
        "sites_tss",
        "Stage I TSS (all novel sites)",
        {"analysis_scope": "sites_all_novel", "entity": "tss"},
        False,
    ),
    (
        "sites_tes",
        "Stage I TES (all novel sites)",
        {"analysis_scope": "sites_all_novel", "entity": "tes"},
        False,
    ),
    (
        "transcript_novel_pr",
        "Transcript PR (novel filtered assembly)",
        {"analysis_scope": "transcript_filtered_assembly"},
        True,
    ),
)

BENCHMARK_RUN_RE = re.compile(
    r"^(?P<data_type>.+?)__train_(?P<train_annotation>.+?)__test_(?P<test_annotation>.+?)"
    r"(?:__(?P<variant>.+))?$"
)

DEFAULT_PLOT_V2_ROOT = Path("plot_v2")

MODEL_BASELINE_LABEL = "Baseline"
MODEL_RF_LABEL = "Telos-RF"
MODEL_XGB_LABEL = "Telos-XGB"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "figure.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def parse_benchmark_run_id(run_id: str) -> tuple[str, str, str, str] | None:
    m = BENCHMARK_RUN_RE.match(run_id.strip())
    if not m:
        return None
    variant = (m.group("variant") or "").strip()
    return (
        m.group("data_type").strip(),
        m.group("train_annotation").strip(),
        m.group("test_annotation").strip(),
        variant,
    )


def load_benchmark_summaries(
    root: Path,
    *,
    row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load all ``benchmark_summary.csv`` files under *root* run folders."""
    rows: list[pd.DataFrame] = []
    for summary in sorted(root.glob("*/reports/benchmark_summary.csv")):
        run_id = summary.parent.parent.name
        parsed = parse_benchmark_run_id(run_id)
        if parsed is None:
            continue
        dt, tr, te, variant = parsed
        df = pd.read_csv(summary)
        if df.empty:
            continue
        df["run_id"] = run_id
        df["data_type"] = dt
        df["train_annotation"] = tr
        df["test_annotation"] = te
        df["run_variant"] = variant
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No benchmark_summary.csv under {root} (expected *__train_*__test_* run folders)."
        )
    out = pd.concat(rows, axis=0, ignore_index=True)
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "ok"].copy()
    if row_filter is not None:
        out = row_filter(out)
    if out.empty:
        raise ValueError("No benchmark summary rows left after filtering.")
    return out


def _extract_metric_scores(
    df: pd.DataFrame,
    spec: BenchmarkMetricSpec,
) -> pd.DataFrame:
    """Per-row baseline / rf / xgb for one benchmark metric."""
    stem, _title, crf, cxgb, cbase, scale_100 = spec
    if not all(c in df.columns for c in (crf, cxgb, cbase)):
        return pd.DataFrame()

    recs: list[dict[str, Any]] = []
    id_cols = [
        c
        for c in (
            "run_id",
            "data_type",
            "assembler_id",
            "train_annotation",
            "test_annotation",
            "test_id",
            "run_variant",
        )
        if c in df.columns
    ]
    for _, r in df.iterrows():
        b = pd.to_numeric(r.get(cbase), errors="coerce")
        rf = pd.to_numeric(r.get(crf), errors="coerce")
        xgb = pd.to_numeric(r.get(cxgb), errors="coerce")
        if pd.isna(b) and pd.isna(rf) and pd.isna(xgb):
            continue
        if scale_100:
            b = b / TRANSCRIPT_PR_AUC_DIVISOR if pd.notna(b) else b
            rf = rf / TRANSCRIPT_PR_AUC_DIVISOR if pd.notna(rf) else rf
            xgb = xgb / TRANSCRIPT_PR_AUC_DIVISOR if pd.notna(xgb) else xgb
        row: dict[str, Any] = {
            "metric_id": stem,
            "baseline": float(b) if pd.notna(b) else np.nan,
            "rf": float(rf) if pd.notna(rf) else np.nan,
            "xgb": float(xgb) if pd.notna(xgb) else np.nan,
        }
        for c in id_cols:
            row[c] = r.get(c, "")
        if "assembler_id" in row:
            row["assembler_id"] = str(row["assembler_id"]).strip().lower() or "unknown"
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def aggregate_benchmark_by_groups(
    scores: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Mean baseline / rf / xgb per *group_cols*."""
    if scores.empty:
        return scores
    cols = [c for c in group_cols if c in scores.columns]
    if not cols:
        cols = ["data_type", "assembler_id"]

    out = (
        scores.groupby(cols, dropna=False)[["baseline", "rf", "xgb"]]
        .mean()
        .reset_index()
    )
    out["n_tests"] = scores.groupby(cols, dropna=False).size().values
    return out


def metric_row_label(metric_id: str, fallback: str = "") -> str:
    return METRIC_ROW_LABEL.get(metric_id, fallback or metric_id)


def assembler_display_name(assembler_id: str) -> str:
    asm = str(assembler_id).strip().lower()
    names = {
        "stringtie": "StringTie",
        "isoquant": "IsoQuant",
        "scallop2": "Scallop2",
    }
    return names.get(asm, asm)


def combo_label(row: pd.Series) -> str:
    """X-axis: assembler only (data-type blocks labeled above the plot)."""
    asm = str(row.get("assembler_id", "")).strip().lower() or "?"
    return assembler_display_name(asm)


DATA_TYPE_ORDER: tuple[str, ...] = ("sr", "cdna", "drna", "pacbio")


def _data_type_sort_key(dt: str) -> int:
    dt = str(dt).strip().lower()
    try:
        return DATA_TYPE_ORDER.index(dt)
    except ValueError:
        return len(DATA_TYPE_ORDER)


def _short_test_label(test_id: str, assembler: str) -> str:
    """Strip ``__<assembler>`` suffix from benchmark test_id for x-axis labels."""
    tid = str(test_id).strip()
    asm = str(assembler).strip().lower()
    suffix = f"__{asm}"
    if tid.lower().endswith(suffix):
        return tid[: -len(suffix)]
    return tid


# Layout: spaced x-axis, thin bar triplets, two-column y labels (AuPR | metric/panel).
_BAR_SPACING_FACTOR = 1.22
_BAR_WIDTH = 0.17
_BAR_FIGSIZE_BASE = (10.0, 6.0)
_BAR_DPI = 300
_BAR_INCHES_PER_XUNIT = 0.38
_STACKED_HSPACE = 0.22
_STACKED_PANEL_H = 3.4
_STACKED_LEFT = 0.14
_AUPR_YLABEL_FIG = "Area Under Precision–Recall Curve (AuPR)"
_AUPR_LABEL_X = 0.028
# Metric/panel column (TSS / TES / Transcript): left of y-tick labels (figure coords, ha=left).
_METRIC_LABEL_INSET = 0.3
_METRIC_LABEL_X = 0.058
_LEGEND_FIG_Y = 0.98
_STACKED_TOP = 0.90
_GROUP_LABEL_Y = 1.015
# v1 shifts "Short Reads" left for wide SR blocks (9 datasets); not for 2–3 aggregated bars.
_SR_GROUP_LABEL_SHIFT_MIN_SIZE = 6
_SR_GROUP_LABEL_SHIFT_X = 2.2 * _BAR_SPACING_FACTOR
_FONT_GROUP_TOP = 13
_FONT_AUPR_LABEL = 15
_FONT_METRIC_LABEL = 15
_FONT_XTICK = 12
_FONT_YTICK = 12
_FONT_LEGEND = 15
_LEGEND_COLUMNSPACING = 2.4
_LEGEND_HANDLETEXTPAD = 1.0

COLOR_BASELINE = "#1f77b4"
COLOR_TELOS_RF = "#ff7f0e"
COLOR_TELOS_XGB = "#2ca02c"

METRIC_ROW_LABEL: dict[str, str] = {
    "stage1_tss": "TSS",
    "stage1_tes": "TES",
    "transcript_pr": "Transcript",
    "transcript_novel_pr": "Transcript",
    "sites_tss": "TSS",
    "sites_tes": "TES",
}

DATA_TYPE_TITLE: dict[str, str] = {
    "sr": "Short Reads",
    "cdna": "cDNA",
    "drna": "dRNA",
    "pacbio": "PacBio",
}

# Consistent modality order on x-axis (matches v1 cross-annotation: PacBio → cDNA → dRNA → SR).
DATA_TYPE_PLOT_ORDER: tuple[str, ...] = ("pacbio", "cdna", "drna", "sr")

# Single-test mouse / tissue benchmarks: one accession per data_type (paper table).
MOUSE_DATASET_ACCESSIONS: dict[str, str] = {
    "cdna": "ENCFF683TBO",
    "drna": "ENCFF765AEC",
    "pacbio": "ENCFF874VSI",
    "sr": "ENCSR982PLD",
}

TISSUE_DATASET_ACCESSIONS: dict[str, str] = {
    "cdna": "SRR31255649",
    "drna": "SRR36400176",
    "pacbio": "ENCFF185VYD",
    "sr": "ENCSR321PGV",
}


def build_dataset_group_titles(accessions: dict[str, str]) -> dict[str, str]:
    """Top-of-plot group labels: accession on line 1, data-type title on line 2."""
    out: dict[str, str] = {}
    for dt, acc in accessions.items():
        key = str(dt).strip().lower()
        type_title = DATA_TYPE_TITLE.get(key, key)
        out[key] = f"{acc}\n{type_title}"
    return out


MOUSE_DATASET_GROUP_TITLES = build_dataset_group_titles(MOUSE_DATASET_ACCESSIONS)
TISSUE_DATASET_GROUP_TITLES = build_dataset_group_titles(TISSUE_DATASET_ACCESSIONS)


def _group_title_for_key(group_key: str) -> str:
    key = str(group_key).strip().lower()
    if key in V1_GROUP_TITLE:
        return V1_GROUP_TITLE[key]
    return DATA_TYPE_TITLE.get(key, str(group_key))


def _plot_group_order_keys(group_col: str) -> tuple[str, ...]:
    if group_col == "v1_group":
        return V1_GROUP_ORDER
    return DATA_TYPE_PLOT_ORDER


def sort_rows_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows PacBio → cDNA → dRNA → SR (then assembler) for every figure."""
    if df.empty:
        return df
    out = df.copy()
    if "v1_group" in out.columns:
        order = {g: i for i, g in enumerate(V1_GROUP_ORDER)}
        out["_ord"] = out["v1_group"].astype(str).str.lower().map(lambda g: order.get(g, 99))
        sort_cols = ["_ord"]
    elif "data_type" in out.columns:
        order = {g: i for i, g in enumerate(DATA_TYPE_PLOT_ORDER)}
        out["_ord"] = out["data_type"].astype(str).str.lower().map(lambda g: order.get(g, 99))
        sort_cols = ["_ord"]
    else:
        return out.reset_index(drop=True)
    if "assembler_id" in out.columns:
        sort_cols.append("assembler_id")
    out = out.sort_values(sort_cols, kind="stable").drop(columns="_ord")
    return out.reset_index(drop=True)


def _add_group_labels_top(
    ax: plt.Axes,
    ordered: pd.DataFrame,
    x_pos: np.ndarray,
    *,
    group_title_overrides: dict[str, str] | None = None,
) -> None:
    """Data-type block titles above the plot (PacBio / cDNA / dRNA / Short Reads)."""
    group_col = "v1_group" if "v1_group" in ordered.columns else "data_type"
    overrides = {str(k).lower(): v for k, v in (group_title_overrides or {}).items()}
    pos = 0
    for grp_key in _plot_group_order_keys(group_col):
        g = ordered[ordered[group_col].astype(str).str.lower() == grp_key]
        size = len(g)
        if size == 0:
            continue
        mid = (float(x_pos[pos]) + float(x_pos[pos + size - 1])) / 2.0
        if str(grp_key) in ("srr", "sr") and size >= _SR_GROUP_LABEL_SHIFT_MIN_SIZE:
            mid -= _SR_GROUP_LABEL_SHIFT_X
        title = overrides.get(str(grp_key).lower()) or _group_title_for_key(str(grp_key))
        ax.text(
            mid,
            _GROUP_LABEL_Y,
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_FONT_GROUP_TOP,
            fontweight="bold",
            clip_on=False,
        )
        pos += size


def _spaced_x_positions(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return np.arange(n, dtype=float) * _BAR_SPACING_FACTOR


def _datatype_group_separators_x(x_pos: np.ndarray, group_sizes: list[int]) -> list[float]:
    """Vertical separators midway between spaced cluster centers (legacy style)."""
    lines: list[float] = []
    pos = 0
    for size in group_sizes[:-1]:
        pos += size
        lines.append((float(x_pos[pos - 1]) + float(x_pos[pos])) / 2.0)
    return lines


def _bar_figsize(n_clusters: int, x_pos: np.ndarray | None = None) -> tuple[float, float]:
    """Figure width tracks the spaced x-axis span so bars fill the canvas."""
    _base_w, base_h = _BAR_FIGSIZE_BASE
    if n_clusters <= 0:
        return _BAR_FIGSIZE_BASE
    x_span = float(x_pos[-1]) if x_pos is not None and len(x_pos) else (n_clusters - 1) * _BAR_SPACING_FACTOR
    w = max(_base_w, min(22.0, _BAR_INCHES_PER_XUNIT * (x_span + 2.0)))
    h = 6.8 if n_clusters > 12 else base_h
    return (w, h)


def _lock_xaxis(ax: plt.Axes, x_pos: np.ndarray) -> None:
    """Prevent tight_layout / savefig from collapsing the spaced x-axis to ~0–1."""
    if len(x_pos) == 0:
        return
    pad = 0.75 * _BAR_SPACING_FACTOR
    ax.set_xlim(float(x_pos[0]) - pad, float(x_pos[-1]) + pad)
    ax.set_autoscale_on(False)
    ax.autoscale(enable=False, axis="x")


def _save_bar_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    x_pos: np.ndarray,
    out_path: Path,
    *,
    has_suptitle: bool = False,
    bottom: float = 0.22,
) -> None:
    for ax in axes:
        _lock_xaxis(ax, x_pos)
    top = 0.90 if not has_suptitle else 0.88
    fig.subplots_adjust(left=0.07, right=0.99, bottom=bottom, top=top)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_BAR_DPI)
    plt.close(fig)


def _figure_legend(fig: plt.Figure) -> None:
    """Legend in the top figure margin (does not overlap PacBio/cDNA headers)."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COLOR_BASELINE, alpha=0.9, label=MODEL_BASELINE_LABEL),
        Patch(facecolor=COLOR_TELOS_RF, alpha=0.9, label=MODEL_RF_LABEL),
        Patch(facecolor=COLOR_TELOS_XGB, alpha=0.9, label=MODEL_XGB_LABEL),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, _LEGEND_FIG_Y),
        bbox_transform=fig.transFigure,
        fontsize=_FONT_LEGEND,
        ncol=3,
        frameon=True,
        columnspacing=_LEGEND_COLUMNSPACING,
        handletextpad=_LEGEND_HANDLETEXTPAD,
        handlelength=1.6,
        borderaxespad=0.0,
    )


def _stacked_bottom_margin(n_clusters: int, *, single_line_xticks: bool = False) -> float:
    """Bottom margin so rotated x tick labels are not clipped."""
    if single_line_xticks:
        if n_clusters > 8:
            return 0.16
        return 0.14
    if n_clusters > 14:
        return 0.30
    if n_clusters > 8:
        return 0.24
    return 0.18


def _add_figure_side_ylabel(fig: plt.Figure, text: str) -> None:
    fig.text(
        _AUPR_LABEL_X,
        0.5,
        text,
        transform=fig.transFigure,
        rotation=90,
        va="center",
        ha="center",
        fontsize=_FONT_AUPR_LABEL,
        linespacing=0.75,
    )


def _add_figure_aupr_ylabel(fig: plt.Figure) -> None:
    _add_figure_side_ylabel(fig, _AUPR_YLABEL_FIG)


def _pdf_output_path(path: Path) -> Path:
    return path.with_suffix(".pdf")


def _aupr_ylim_max(base_v: np.ndarray, rf_v: np.ndarray, xgb_v: np.ndarray) -> float:
    vals = np.concatenate([base_v, rf_v, xgb_v])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return float(vals.max())*1.1


def _add_metric_row_label_fig(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    """Second y-label column (TSS / assembler panel), just right of the y-axis."""
    if not text.strip():
        return
    pos = ax.get_position()
    fig.text(
        _METRIC_LABEL_X,
        pos.y0 + pos.height * 0.5,
        text.strip(),
        transform=fig.transFigure,
        rotation=90,
        va="center",
        ha="left",
        fontsize=_FONT_METRIC_LABEL,
        fontweight="bold"
    )


def _plot_three_bars_at_positions(
    ax: plt.Axes,
    x_pos: np.ndarray,
    base_v: np.ndarray,
    rf_v: np.ndarray,
    xgb_v: np.ndarray,
) -> None:
    """Draw baseline / Telos-RF / Telos-XGB clusters at spaced x positions."""
    n = len(x_pos)
    categories = [
        (base_v, COLOR_BASELINE, 0.9),
        (rf_v, COLOR_TELOS_RF, 0.9),
        (xgb_v, COLOR_TELOS_XGB, 0.9),
    ]
    n_cat = len(categories)
    for idx, (values, color, alpha) in enumerate(categories):
        offsets = [
            float(x_pos[i]) + (idx - (n_cat - 1) / 2) * _BAR_WIDTH for i in range(n)
        ]
        ax.bar(offsets, values, width=_BAR_WIDTH, color=color, alpha=alpha)


@dataclass(frozen=True)
class BarPanel:
    """One row in a vertically stacked bar figure."""

    row_label: str
    labels: list[str]
    base_v: np.ndarray
    rf_v: np.ndarray
    xgb_v: np.ndarray
    ordered: pd.DataFrame | None = None
    group_title_overrides: dict[str, str] | None = None


def _decorate_panel_groups(
    ax: plt.Axes,
    ordered: pd.DataFrame,
    x_pos: np.ndarray,
    *,
    draw_group_titles: bool,
    group_title_overrides: dict[str, str] | None = None,
) -> None:
    group_col = "v1_group" if "v1_group" in ordered.columns else "data_type"
    group_sizes = [
        len(ordered[ordered[group_col].astype(str).str.lower() == k])
        for k in _plot_group_order_keys(group_col)
        if len(ordered[ordered[group_col].astype(str).str.lower() == k]) > 0
    ]
    for sep_x in _datatype_group_separators_x(x_pos, group_sizes):
        ax.axvline(sep_x, color="#888888", linestyle="--", linewidth=1.5, alpha=0.85, zorder=0)
    if draw_group_titles:
        _add_group_labels_top(
            ax, ordered, x_pos, group_title_overrides=group_title_overrides
        )


def _render_panel_on_ax(
    ax: plt.Axes,
    x_pos: np.ndarray,
    panel: BarPanel,
    *,
    show_xticklabels: bool,
    draw_group_titles: bool = False,
    ylim_max: float | None = None,
) -> None:
    _plot_three_bars_at_positions(ax, x_pos, panel.base_v, panel.rf_v, panel.xgb_v)
    if panel.ordered is not None and not panel.ordered.empty:
        _decorate_panel_groups(
            ax,
            panel.ordered,
            x_pos,
            draw_group_titles=draw_group_titles,
            group_title_overrides=panel.group_title_overrides,
        )
    ymax = float(ylim_max) if ylim_max is not None else _aupr_ylim_max(
        panel.base_v, panel.rf_v, panel.xgb_v
    )
    ax.set_ylim(0.0, ymax)
    _lock_xaxis(ax, x_pos)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=_FONT_YTICK)
    ax.grid(axis="y", linestyle=":", alpha=0.3, linewidth=0.6)
    if show_xticklabels:
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(panel.labels, rotation=45, ha="right", fontsize=_FONT_XTICK)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def draw_stacked_bar_panels(
    panels: list[BarPanel],
    out_path: Path,
    *,
    show_aupr_ylabel: bool = True,
    figure_ylabel: str | None = None,
) -> None:
    """Vertically stacked panels; two-column y labels; saves PDF."""
    if not panels:
        return
    n = len(panels)
    n_clusters = len(panels[0].labels)
    x_pos = _spaced_x_positions(n_clusters)
    fig_w = _bar_figsize(n_clusters, x_pos)[0]
    fig_h = _STACKED_PANEL_H * n + 0.6
    fig, axes_arr = plt.subplots(
        n, 1, figsize=(fig_w, fig_h), dpi=_BAR_DPI, sharex=True, constrained_layout=False
    )
    axes = [axes_arr] if n == 1 else list(axes_arr)

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        _render_panel_on_ax(
            ax,
            x_pos,
            panel,
            show_xticklabels=(i == n - 1),
            draw_group_titles=(i == 0 and panel.ordered is not None and not panel.ordered.empty),
        )

    last_labels = panels[-1].labels if panels else []
    single_line_xticks = bool(last_labels) and all("\n" not in lb for lb in last_labels)
    bottom = _stacked_bottom_margin(n_clusters, single_line_xticks=single_line_xticks)
    for ax in axes:
        _lock_xaxis(ax, x_pos)
    fig.subplots_adjust(
        left=_STACKED_LEFT,
        right=0.99,
        bottom=bottom,
        top=_STACKED_TOP,
        hspace=_STACKED_HSPACE,
    )

    if figure_ylabel:
        _add_figure_side_ylabel(fig, figure_ylabel)
    elif show_aupr_ylabel:
        _add_figure_aupr_ylabel(fig)
    for ax, panel in zip(axes, panels):
        _add_metric_row_label_fig(fig, ax, panel.row_label)
    _figure_legend(fig)
    pdf_path = _pdf_output_path(out_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    plt.close(fig)


def order_scores_for_v1_tool_panel(scores: pd.DataFrame, *, panel: str) -> pd.DataFrame:
    """Filter v1 tool panel and sort 18 canonical datasets (9 SR + 9 long-read)."""
    mask = scores.apply(lambda r: row_matches_tool_panel(r, panel), axis=1)
    sub = scores.loc[mask].copy()
    if sub.empty:
        return sub
    return annotate_v1_labels(sub)


def draw_grouped_bars(
    labels: list[str],
    base_v: np.ndarray,
    rf_v: np.ndarray,
    xgb_v: np.ndarray,
    title: str,
    out_path: Path,
    *,
    ylim_max: float | None = None,
) -> None:
    """Single-panel grouped bars (prefer ``draw_stacked_bar_panels`` for multi-metric output)."""
    draw_stacked_bar_panels(
        [
            BarPanel(
                row_label="",
                labels=labels,
                base_v=base_v,
                rf_v=rf_v,
                xgb_v=xgb_v,
            )
        ],
        out_path,
    )


def plot_cross_annotation_per_dataset_bars(
    df: pd.DataFrame,
    outdir: Path,
    *,
    panels: tuple[str, ...] = CROSS_ANNOTATION_TOOL_PANELS,
    metric_specs: tuple[BenchmarkMetricSpec, ...] = CORE_BENCHMARK_METRICS,
    long_csv_name: str = "aupr_per_dataset_long.csv",
) -> None:
    """
    Cross-annotation per-dataset bars: one stacked PDF per assembler panel.

    Each figure stacks TSS, TES, and Transcript (18 datasets per row).
    Outputs ``aupr_bars_stringtie.pdf`` and ``aupr_bars_other.pdf``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("aupr_bars_stage1_*.pdf"):
        stale.unlink(missing_ok=True)
    stale_tx = outdir / "aupr_bars_transcript_pr.pdf"
    if stale_tx.is_file():
        stale_tx.unlink()
    long_rows: list[dict[str, Any]] = []
    n_exp = expected_panel_size()

    for panel in panels:
        stack_panels: list[BarPanel] = []

        for spec in metric_specs:
            stem, title, _crf, _cxgb, _cbase, _scale = spec
            per_row = _extract_metric_scores(df, spec)
            if per_row.empty:
                print(f"[telos_repro] cross-annotation plot: skip {stem}/{panel} (no data)")
                continue

            ordered = order_scores_for_v1_tool_panel(per_row, panel=panel)
            if ordered.empty:
                print(f"[telos_repro] cross-annotation plot: skip {stem}/{panel} (no rows)")
                continue
            if len(ordered) != n_exp:
                print(
                    f"[telos_repro] cross-annotation plot: {stem}/{panel} has {len(ordered)} "
                    f"rows (expected {n_exp})"
                )

            ordered = ordered.copy()
            ordered["plot_panel"] = panel
            ordered.to_csv(outdir / f"aupr_per_dataset_{stem}__{panel}.tsv", sep="\t", index=False)

            stack_panels.append(
                BarPanel(
                    row_label=metric_row_label(stem, title),
                    labels=ordered["dataset_label"].tolist(),
                    base_v=ordered["baseline"].to_numpy(dtype=float),
                    rf_v=ordered["rf"].to_numpy(dtype=float),
                    xgb_v=ordered["xgb"].to_numpy(dtype=float),
                    ordered=ordered,
                )
            )

            for _, r in ordered.iterrows():
                for model, col in (("baseline", "baseline"), ("rf", "rf"), ("xgb", "xgb")):
                    v = pd.to_numeric(r.get(col), errors="coerce")
                    if pd.isna(v):
                        continue
                    long_rows.append(
                        {
                            "metric_id": stem,
                            "plot_panel": panel,
                            "assembler_id": r.get("assembler_id", ""),
                            "data_type": r.get("data_type", ""),
                            "v1_group": r.get("v1_group", ""),
                            "v1_dataset_key": r.get("v1_dataset_key", ""),
                            "test_id": r.get("test_id", ""),
                            "dataset_label": r.get("dataset_label", ""),
                            "model": model,
                            "value": float(v),
                        }
                    )

        if stack_panels:
            draw_stacked_bar_panels(stack_panels, outdir / f"aupr_bars_{panel}.pdf")

    if long_rows:
        pd.DataFrame.from_records(long_rows).to_csv(
            outdir / long_csv_name, index=False
        )


def _melt_agg_long(
    agg: pd.DataFrame,
    metric_id: str,
    metric_title: str,
    extra_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    facet_cols = [c for c in extra_cols if c in agg.columns]
    for _, r in agg.iterrows():
        for model, col in (("baseline", "baseline"), ("rf", "rf"), ("xgb", "xgb")):
            v = pd.to_numeric(r.get(col), errors="coerce")
            if pd.isna(v):
                continue
            rec: dict[str, Any] = {
                "metric_id": metric_id,
                "metric_title": metric_title,
                "data_type": r.get("data_type", ""),
                "assembler_id": r.get("assembler_id", ""),
                "model": model,
                "value": float(v),
                "n_tests": int(r.get("n_tests", 0)),
            }
            for fc in facet_cols:
                rec[fc] = r.get(fc, "")
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def plot_benchmark_aupr_bars(
    df: pd.DataFrame,
    outdir: Path,
    *,
    group_cols: list[str] | None = None,
    metric_specs: tuple[BenchmarkMetricSpec, ...] = CORE_BENCHMARK_METRICS,
    file_prefix: str = "",
    long_csv_name: str = "aupr_long.csv",
    group_title_overrides: dict[str, str] | None = None,
) -> None:
    """Aggregate benchmark summaries; one stacked PNG (TSS / TES / transcript PR)."""
    outdir.mkdir(parents=True, exist_ok=True)
    groups = group_cols or ["data_type", "assembler_id"]
    long_parts: list[pd.DataFrame] = []
    stack_panels: list[BarPanel] = []
    prefix = f"{file_prefix}_" if file_prefix else ""

    for spec in metric_specs:
        stem, title, _crf, _cxgb, _cbase, _scale = spec
        per_row = _extract_metric_scores(df, spec)
        if per_row.empty:
            print(f"[telos_repro] plot: skip {stem} (no data)")
            continue
        agg = aggregate_benchmark_by_groups(per_row, groups)
        if agg.empty:
            continue
        agg = sort_rows_for_plot(agg)
        agg.to_csv(outdir / f"{prefix}aupr_agg_{stem}.tsv", sep="\t", index=False)
        long_parts.append(_melt_agg_long(agg, stem, title, tuple(groups)))

        labels = [combo_label(r) for _, r in agg.iterrows()]
        stack_panels.append(
            BarPanel(
                row_label=metric_row_label(stem, title),
                labels=labels,
                base_v=agg["baseline"].to_numpy(dtype=float),
                rf_v=agg["rf"].to_numpy(dtype=float),
                xgb_v=agg["xgb"].to_numpy(dtype=float),
                ordered=agg,
                group_title_overrides=group_title_overrides,
            )
        )

    if stack_panels:
        draw_stacked_bar_panels(stack_panels, outdir / f"{prefix}aupr_bars_stacked.pdf")

    if long_parts:
        pd.concat(long_parts, axis=0, ignore_index=True).to_csv(
            outdir / long_csv_name, index=False
        )


# --- RefSeq-novel eval adapter ---


def load_novel_summary(reports: Path) -> pd.DataFrame:
    """Load combined summary or merge per-metric TSVs from the eval script."""
    reports = reports.resolve()
    combined = reports / "novel_eval_summary.tsv"
    if combined.is_file():
        return pd.read_csv(combined, sep="\t")
    tx_path = reports / "novel_transcript_pr_by_test.tsv"
    site_path = reports / "novel_stage1_by_test.tsv"
    parts: list[pd.DataFrame] = []
    if tx_path.is_file():
        parts.append(pd.read_csv(tx_path, sep="\t"))
    if site_path.is_file():
        parts.append(pd.read_csv(site_path, sep="\t"))
    if not parts:
        raise FileNotFoundError(
            f"No novel eval outputs under {reports} "
            "(expected novel_eval_summary.tsv or transcript/stage1 TSVs)."
        )
    return pd.concat(parts, axis=0, ignore_index=True, sort=False)


def _filter_novel_metric_rows(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    sub = df.copy()
    for col, val in filters.items():
        if col not in sub.columns:
            return sub.iloc[0:0]
        sub = sub[sub[col].astype(str) == str(val)]
    if "pr_status" in sub.columns and "analysis_scope" in sub.columns:
        tx = sub["analysis_scope"].astype(str) == "transcript_filtered_assembly"
        sub = sub[~tx | sub["pr_status"].astype(str).str.lower().eq("ok")]
    return sub


def _collapse_novel_test_to_scores(sub: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["data_type", "assembler_id", "test_id"]
    if "test_annotation" in sub.columns:
        group_cols.append("test_annotation")
    if "cell_id" in sub.columns:
        group_cols.append("cell_id")

    recs: list[dict[str, Any]] = []
    for keys, g in sub.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_cols, keys))
        base = pd.to_numeric(g["auc_baseline"], errors="coerce").dropna()
        rf = pd.to_numeric(g.loc[g["model"].astype(str) == "rf", "auc_model"], errors="coerce")
        xgb = pd.to_numeric(g.loc[g["model"].astype(str) == "xgb", "auc_model"], errors="coerce")
        row["baseline"] = float(base.iloc[0]) if len(base) else np.nan
        row["rf"] = float(rf.iloc[0]) if len(rf) else np.nan
        row["xgb"] = float(xgb.iloc[0]) if len(xgb) else np.nan
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def _extract_novel_metric_scores(
    summary: pd.DataFrame,
    filters: dict[str, Any],
    *,
    facet_col: str | None = None,
    facet_val: str | None = None,
    scale_100: bool = False,
) -> pd.DataFrame:
    """Per-test baseline / rf / xgb for one novel-eval metric."""
    sub = _filter_novel_metric_rows(summary, filters)
    if facet_val is not None and facet_col and facet_col in sub.columns:
        sub = sub[sub[facet_col].astype(str) == facet_val]
    if sub.empty:
        return sub
    scores = _collapse_novel_test_to_scores(sub)
    if scores.empty:
        return scores
    if scale_100:
        for c in ("baseline", "rf", "xgb"):
            scores[c] = pd.to_numeric(scores[c], errors="coerce") / TRANSCRIPT_PR_AUC_DIVISOR
    scores["assembler_id"] = scores["assembler_id"].astype(str).str.strip().str.lower()
    scores["data_type"] = scores["data_type"].astype(str).str.strip().str.lower()
    return scores


def aggregate_by_datatype_assembler(
    df: pd.DataFrame,
    *,
    facet_col: str | None = None,
) -> pd.DataFrame:
    agg_cols = ["data_type", "assembler_id"]
    if facet_col and facet_col in df.columns:
        agg_cols.append(facet_col)

    collapsed = _collapse_novel_test_to_scores(df)
    if collapsed.empty:
        return collapsed

    collapsed["assembler_id"] = collapsed["assembler_id"].astype(str).str.strip().str.lower()
    collapsed["data_type"] = collapsed["data_type"].astype(str).str.strip()

    out = (
        collapsed.groupby(agg_cols, dropna=False)[["baseline", "rf", "xgb"]]
        .mean()
        .reset_index()
    )
    out["n_tests"] = collapsed.groupby(agg_cols, dropna=False).size().values
    return out


def plot_novel_per_dataset_bars(
    summary: pd.DataFrame,
    outdir: Path,
    *,
    facet_col: str | None = "test_annotation",
    panels: tuple[str, ...] = CROSS_ANNOTATION_TOOL_PANELS,
    metric_specs: tuple[NovelMetricSpec, ...] = NOVEL_METRIC_SPECS,
    long_csv_name: str = "refseq_novel_aupr_per_dataset_long.csv",
) -> None:
    """
    Refseq-novel per-dataset bars: one stacked PDF per assembler panel.

    Each figure stacks transcript PR, TSS, and TES (18 datasets per row).
    Outputs ``aupr_bars_stringtie.pdf`` and ``aupr_bars_other.pdf`` per facet.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if facet_col and facet_col in summary.columns:
        facets: list[str | None] = sorted(summary[facet_col].astype(str).unique())
    else:
        facets = [None]

    for facet_val in facets:
        facet_out = outdir / str(facet_val) if facet_val is not None else outdir
        facet_out.mkdir(parents=True, exist_ok=True)
        for stale in facet_out.glob("aupr_bars_stacked*.pdf"):
            stale.unlink(missing_ok=True)

        long_rows: list[dict[str, Any]] = []
        n_exp = expected_panel_size()

        for panel in panels:
            stack_panels: list[BarPanel] = []

            for stem, title, filters, scale_100 in metric_specs:
                per_row = _extract_novel_metric_scores(
                    summary,
                    filters,
                    facet_col=facet_col if facet_val else None,
                    facet_val=facet_val,
                    scale_100=scale_100,
                )
                if per_row.empty:
                    print(f"[telos_repro] refseq-novel plot: skip {stem}/{panel} (no data)")
                    continue

                ordered = order_scores_for_v1_tool_panel(per_row, panel=panel)
                if ordered.empty:
                    print(f"[telos_repro] refseq-novel plot: skip {stem}/{panel} (no rows)")
                    continue
                if len(ordered) != n_exp:
                    print(
                        f"[telos_repro] refseq-novel plot: {stem}/{panel} has {len(ordered)} "
                        f"rows (expected {n_exp})"
                    )

                ordered = ordered.copy()
                ordered["plot_panel"] = panel
                if facet_val is not None and facet_col:
                    ordered[facet_col] = facet_val
                ordered.to_csv(
                    facet_out / f"aupr_per_dataset_{stem}__{panel}.tsv", sep="\t", index=False
                )

                stack_panels.append(
                    BarPanel(
                        row_label=metric_row_label(stem, title),
                        labels=ordered["dataset_label"].tolist(),
                        base_v=ordered["baseline"].to_numpy(dtype=float),
                        rf_v=ordered["rf"].to_numpy(dtype=float),
                        xgb_v=ordered["xgb"].to_numpy(dtype=float),
                        ordered=ordered,
                    )
                )

                for _, r in ordered.iterrows():
                    for model, col in (("baseline", "baseline"), ("rf", "rf"), ("xgb", "xgb")):
                        v = pd.to_numeric(r.get(col), errors="coerce")
                        if pd.isna(v):
                            continue
                        rec: dict[str, Any] = {
                            "metric_id": stem,
                            "plot_panel": panel,
                            "assembler_id": r.get("assembler_id", ""),
                            "data_type": r.get("data_type", ""),
                            "v1_group": r.get("v1_group", ""),
                            "v1_dataset_key": r.get("v1_dataset_key", ""),
                            "test_id": r.get("test_id", ""),
                            "dataset_label": r.get("dataset_label", ""),
                            "model": model,
                            "value": float(v),
                        }
                        if facet_val is not None and facet_col:
                            rec[facet_col] = facet_val
                        long_rows.append(rec)

            if stack_panels:
                draw_stacked_bar_panels(stack_panels, facet_out / f"aupr_bars_{panel}.pdf")

        if long_rows:
            csv_name = long_csv_name
            if facet_val is not None and facet_col:
                stem, suffix = long_csv_name.rsplit(".", 1)
                csv_name = f"{stem}__{facet_val}.{suffix}"
            pd.DataFrame.from_records(long_rows).to_csv(facet_out / csv_name, index=False)


def plot_novel_aupr_bars(
    summary: pd.DataFrame,
    outdir: Path,
    *,
    facet_col: str | None = "test_annotation",
    metric_specs: tuple[NovelMetricSpec, ...] = NOVEL_METRIC_SPECS,
) -> None:
    """Refseq-novel eval: mean AUPR bars aggregated by data_type / assembler (legacy)."""
    outdir.mkdir(parents=True, exist_ok=True)
    long_parts: list[pd.DataFrame] = []

    if facet_col and facet_col in summary.columns:
        facets = sorted(summary[facet_col].astype(str).unique())
    else:
        facets = [None]

    for facet_val in facets:
        stack_panels: list[BarPanel] = []

        for stem, title, filters, scale_100 in metric_specs:
            sub = _filter_novel_metric_rows(summary, filters)
            if facet_val is not None and facet_col in sub.columns:
                sub = sub[sub[facet_col].astype(str) == facet_val]
            if sub.empty:
                print(f"[telos_repro] refseq-novel plot: skip {stem} (facet={facet_val})")
                continue

            agg = aggregate_by_datatype_assembler(
                sub, facet_col=facet_col if facet_val else None
            )
            if agg.empty:
                continue

            if scale_100:
                for c in ("baseline", "rf", "xgb"):
                    agg[c] = pd.to_numeric(agg[c], errors="coerce") / TRANSCRIPT_PR_AUC_DIVISOR

            agg = sort_rows_for_plot(agg)
            suffix = f"__{facet_col}_{facet_val}" if facet_val is not None else ""
            agg.to_csv(outdir / f"aupr_agg_{stem}{suffix}.tsv", sep="\t", index=False)
            long_parts.append(_melt_agg_long(agg, stem, title, (facet_col,) if facet_val else ()))

            labels = [combo_label(r) for _, r in agg.iterrows()]
            stack_panels.append(
                BarPanel(
                    row_label=metric_row_label(stem, title),
                    labels=labels,
                    base_v=agg["baseline"].to_numpy(dtype=float),
                    rf_v=agg["rf"].to_numpy(dtype=float),
                    xgb_v=agg["xgb"].to_numpy(dtype=float),
                    ordered=agg,
                )
            )

        if stack_panels:
            out_name = (
                "aupr_bars_stacked.pdf"
                if facet_val is None
                else f"aupr_bars_stacked__{facet_val}.pdf"
            )
            draw_stacked_bar_panels(stack_panels, outdir / out_name)

    if long_parts:
        pd.concat(long_parts, axis=0, ignore_index=True).to_csv(
            outdir / "refseq_novel_aupr_long.csv", index=False
        )
