"""
Plot publication-style stacked PR-curve figures from benchmark prediction artifacts.

Reads per-test curve point TSVs (Stage I) and gtfcuff PR tables (transcript), then writes
stacked PDF grids matching paper figures 3–8 and 10–12 (gencode train/test by default).

Usage::

  PYTHONPATH=src python src/experiments/plot_pr_curves_from_predictions.py \\
    --root runs/cross_annotation_repro \\
    --train-annotation gencode --test-annotation gencode
"""

from __future__ import annotations

import argparse
import colorsys
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve

from telos_repro.plotting.grouped_aupr_bars import (
    CORE_BENCHMARK_METRICS,
    TRANSCRIPT_PR_AUC_DIVISOR,
    BarPanel,
    _extract_metric_scores,
    apply_plot_style,
    draw_stacked_bar_panels,
    load_benchmark_summaries,
    order_scores_for_v1_tool_panel,
)
from telos_repro.plotting.v1_style_labels import (
    CROSS_ANNOTATION_TOOL_PANELS,
    V1_CANONICAL_DATASET_ORDER,
    annotate_v1_labels,
    pretty_dataset_label,
    v1_dataset_key,
    v1_group_for_datatype,
)

RUN_RE = re.compile(
    r"^(?P<data_type>[^_]+)__train_(?P<train>[^_]+)__test_(?P<test>[^_]+)"
    r"(?:__(?P<variant>.+))?$"
)

RECALL_XLIM_PAD = 0.005
JACCARD_MAX_DIST_BP = 50
JACCARD_SCORE_THRESHOLD = 0.5

# v1 PR-curve styling (generate_stage1_pr_curve.py)
ASSEMBLER_BASE_COLORS: dict[str, str] = {
    "stringtie": "#1f77b4",
    "isoquant": "#ff7f0e",
    "scallop2": "#ff7f0e",
}
MODEL_SHADE_FACTORS: dict[str, float] = {"xgb": 0.75, "rf": 1.0}
BASELINE_SHADE_FACTOR = 0.50
BASELINE_LINESTYLE = ":"
# Distinct linestyles so overlapping RF/XGB curves remain identifiable (color = assembler).
MODEL_LINESTYLES: dict[str, str] = {"rf": "-", "xgb": "-."}
ASSEMBLER_DISPLAY: dict[str, str] = {
    "stringtie": "StringTie",
    "isoquant": "IsoQuant",
    "scallop2": "Scallop2",
}
MODEL_LEGEND_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "rf": "Telos-RF",
    "xgb": "Telos-XGB",
}

LONG_READ_TOOLS: tuple[str, ...] = ("stringtie", "isoquant")
SHORT_READ_TOOLS: tuple[str, ...] = ("stringtie", "scallop2")

LONG_READ_DATASETS: tuple[str, ...] = V1_CANONICAL_DATASET_ORDER[:9]
SHORT_READ_DATASETS: tuple[str, ...] = V1_CANONICAL_DATASET_ORDER[9:]

_SQUARE_PANEL_IN = 3.15
_SQUARE_DPI = 300
_GRID_HSPACE = 0.32
_GRID_WSPACE = 0.28
_GRID_TOP = 0.86
_GRID_BOTTOM = 0.08
_LEGEND_FIG_Y = 0.98
_LEGEND_FONTSIZE = 10
_LEGEND_NCOL = 3


@dataclass(frozen=True)
class StackedPrFigureSpec:
    """One stacked multi-panel PR figure."""

    figure_id: str
    filename: str
    metric: str  # tss | tes | transcript
    datasets: tuple[str, ...]
    tools: tuple[str, ...]
    ncols: int
    nrows: int
    row_group_titles: tuple[str, ...] | None = None  # modality headers per row


PUBLICATION_PR_FIGURES: tuple[StackedPrFigureSpec, ...] = (
    StackedPrFigureSpec(
        "fig03",
        "fig03_stage1_tss_pr_long_reads.pdf",
        "tss",
        LONG_READ_DATASETS,
        LONG_READ_TOOLS,
        3,
        3,
        ("PacBio", "cDNA", "dRNA"),
    ),
    StackedPrFigureSpec(
        "fig04",
        "fig04_stage1_tes_pr_long_reads.pdf",
        "tes",
        LONG_READ_DATASETS,
        LONG_READ_TOOLS,
        3,
        3,
        ("PacBio", "cDNA", "dRNA"),
    ),
    StackedPrFigureSpec(
        "fig05",
        "fig05_stage1_tss_pr_short_reads.pdf",
        "tss",
        SHORT_READ_DATASETS,
        SHORT_READ_TOOLS,
        3,
        3,
        None,
    ),
    StackedPrFigureSpec(
        "fig06",
        "fig06_stage1_tes_pr_short_reads.pdf",
        "tes",
        SHORT_READ_DATASETS,
        SHORT_READ_TOOLS,
        3,
        3,
        None,
    ),
    StackedPrFigureSpec(
        "fig10",
        "fig10_transcript_pr_long_reads.pdf",
        "transcript",
        LONG_READ_DATASETS,
        LONG_READ_TOOLS,
        3,
        3,
        ("PacBio", "cDNA", "dRNA"),
    ),
    StackedPrFigureSpec(
        "fig11",
        "fig11_transcript_pr_short_reads.pdf",
        "transcript",
        SHORT_READ_DATASETS,
        SHORT_READ_TOOLS,
        3,
        3,
        None,
    )
)


def shade_hex_color(hex_color: str, lightness_factor: float) -> str:
    r, g, b = to_rgb(hex_color)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.0, min(1.0, lightness * lightness_factor))
    nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
    return to_hex((nr, ng, nb))


def _scale_transcript_pr_tuple(
    rec: np.ndarray, pre: np.ndarray, ap: float
) -> tuple[np.ndarray, np.ndarray, float]:
    # gtfcuff PR TSV points are already recall/precision in 0–1; summary AUC is percent² units.
    return (
        np.asarray(rec, dtype=float),
        np.asarray(pre, dtype=float),
        float(ap) / TRANSCRIPT_PR_AUC_DIVISOR,
    )


def _recall_xlim_upper(*recall_arrays: np.ndarray) -> float:
    mx = 0.0
    for r in recall_arrays:
        if r is None or len(r) == 0:
            continue
        v = float(np.nanmax(np.asarray(r, dtype=float)))
        if np.isfinite(v):
            mx = max(mx, v)
    upper = mx + RECALL_XLIM_PAD
    if not np.isfinite(upper) or upper <= 0.0:
        return 1.0
    return float(min(1.0, upper))


def _parse_run_id(run_id: str) -> tuple[str, str, str, str] | None:
    m = RUN_RE.match(run_id.strip())
    if not m:
        return None
    return m.group("data_type"), m.group("train"), m.group("test"), (m.group("variant") or "")


def _assembler_from_test_id(test_id: str) -> str:
    tid = str(test_id).strip()
    if "__" in tid:
        return tid.rsplit("__", 1)[-1].strip().lower()
    return ""


def _safe_pr(y: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray, float] | None:
    yt = pd.to_numeric(y, errors="coerce")
    sc = pd.to_numeric(score, errors="coerce")
    keep = yt.notna() & sc.notna()
    if not bool(keep.any()):
        return None
    yv = yt[keep].astype(int).to_numpy()
    sv = sc[keep].astype(float).to_numpy()
    if len(np.unique(yv)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(yv, sv)
    ap = float(average_precision_score(yv, sv))
    return recall, precision, ap


def _curves_from_points_tsv(
    points_tsv: Path,
    *,
    novel_only: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]] | None:
    if not points_tsv.is_file():
        return None
    df = pd.read_csv(points_tsv, sep="\t", low_memory=False)
    need = {"label", "score_rf", "score_xgb", "score_baseline_cov"}
    if not need.issubset(df.columns) or df.empty:
        return None
    if novel_only:
        if "is_novel" not in df.columns:
            return None
        df = df[pd.to_numeric(df["is_novel"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if df.empty:
            return None
    out: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for model, col in (("rf", "score_rf"), ("xgb", "score_xgb"), ("baseline", "score_baseline_cov")):
        pr = _safe_pr(df["label"], df[col])
        if pr is not None:
            out[model] = pr
    return out or None


def _transcript_curves_from_gtfcuff_tables(pr_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, float]] | None:
    rf_curve = pr_dir / "transcript_pr_rf_model_pr.tsv"
    xgb_curve = pr_dir / "transcript_pr_xgb_model_pr.tsv"
    base_curve_rf = pr_dir / "transcript_pr_rf_baseline_pr.tsv"
    base_curve_xgb = pr_dir / "transcript_pr_xgb_baseline_pr.tsv"
    rf_summary = pr_dir / "transcript_pr_rf_aupr_summary.csv"
    xgb_summary = pr_dir / "transcript_pr_xgb_aupr_summary.csv"
    if not (rf_curve.is_file() and xgb_curve.is_file()):
        return None

    def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        if not path.is_file():
            return None
        df = pd.read_csv(path, sep="\t")
        if not {"recall", "precision"}.issubset(df.columns) or df.empty:
            return None
        r = pd.to_numeric(df["recall"], errors="coerce").dropna().to_numpy(dtype=float)
        p = pd.to_numeric(df["precision"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(p) == 0:
            return None
        n = min(len(r), len(p))
        return r[:n], p[:n]

    def _read_ap(summary_csv: Path, curve_name: str) -> float | None:
        if not summary_csv.is_file():
            return None
        df = pd.read_csv(summary_csv)
        if not {"curve", "transcript_pr_auc"}.issubset(df.columns):
            return None
        sub = df[df["curve"].astype(str) == curve_name]
        if sub.empty:
            return None
        v = pd.to_numeric(sub.iloc[0]["transcript_pr_auc"], errors="coerce")
        return float(v) if pd.notna(v) else None

    out: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    rf = _read_curve(rf_curve)
    xgb = _read_curve(xgb_curve)
    if rf is not None:
        ap = _read_ap(rf_summary, "model")
        out["rf"] = _scale_transcript_pr_tuple(rf[0], rf[1], ap or 0.0)
    if xgb is not None:
        ap = _read_ap(xgb_summary, "model")
        out["xgb"] = _scale_transcript_pr_tuple(xgb[0], xgb[1], ap or 0.0)
    bcurve = _read_curve(base_curve_rf) or _read_curve(base_curve_xgb)
    if bcurve is not None:
        bap = _read_ap(rf_summary, "baseline")
        if bap is None:
            bap = _read_ap(xgb_summary, "baseline")
        out["baseline"] = _scale_transcript_pr_tuple(bcurve[0], bcurve[1], bap or 0.0)
    return out or None


def _transcript_curves_from_points_tsv(
    points_tsv: Path,
    *,
    novel_only: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]] | None:
    return _curves_from_points_tsv(points_tsv, novel_only=novel_only)


CurveKey = tuple[str, str, str, str]  # subset, v1_dataset_key, assembler, metric


def _load_curve_index(
    root: Path,
    *,
    train_annotation: str,
    test_annotation: str,
    subset: str,
) -> dict[CurveKey, dict[str, tuple[np.ndarray, np.ndarray, float]]]:
    """Map (subset, dataset, assembler, metric) -> {rf,xgb,baseline} PR tuples."""
    index: dict[CurveKey, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    summary_paths = sorted(root.glob("*/reports/benchmark_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No benchmark_summary.csv found under {root}")

    subsets = [subset] if subset in {"all", "novel"} else ["all", "novel"]
    for summary_csv in summary_paths:
        run_dir = summary_csv.parent.parent
        run_id = run_dir.name
        parsed = _parse_run_id(run_id)
        if parsed is None:
            continue
        data_type, train_ann, test_ann, _variant = parsed
        if train_ann != train_annotation or test_ann != test_annotation:
            continue
        df = pd.read_csv(summary_csv)
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        for _, r in df.iterrows():
            test_id = str(r.get("test_id", "")).strip()
            if not test_id:
                continue
            asm = _assembler_from_test_id(test_id)
            if not asm:
                asm = str(r.get("assembler_id", "")).strip().lower()
            ds_key = v1_dataset_key(data_type, test_id)
            pred_out_raw = str(r.get("predict_outdir", "")).strip()
            pred_out = Path(pred_out_raw) if pred_out_raw else (run_dir / "tests" / test_id)
            curves_dir = pred_out / "reports" / "curves"
            for subset_name in subsets:
                novel_only = subset_name == "novel"
                curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
                tss = _curves_from_points_tsv(curves_dir / "stage1_tss_curve_points.tsv", novel_only=novel_only)
                if tss:
                    curves["tss"] = tss
                tes = _curves_from_points_tsv(curves_dir / "stage1_tes_curve_points.tsv", novel_only=novel_only)
                if tes:
                    curves["tes"] = tes
                if novel_only:
                    tx = _transcript_curves_from_points_tsv(
                        curves_dir / "stage2_tx_curve_points.tsv",
                        novel_only=True,
                    )
                else:
                    tx = _transcript_curves_from_gtfcuff_tables(pred_out / "reports" / "pr")
                if tx:
                    curves["transcript"] = tx
                for metric, model_map in curves.items():
                    key: CurveKey = (subset_name, ds_key, asm, metric)
                    index[key] = model_map
    return index


def _curve_plot_style(assembler: str, model: str) -> tuple[str, str, float]:
    """Return (color, linestyle, linewidth) for one assembler × model curve."""
    base_color = ASSEMBLER_BASE_COLORS.get(assembler, "#333333")
    if model == "baseline":
        return shade_hex_color(base_color, BASELINE_SHADE_FACTOR), BASELINE_LINESTYLE, 2.0
    return (
        shade_hex_color(base_color, MODEL_SHADE_FACTORS[model]),
        MODEL_LINESTYLES.get(model, "-"),
        2.2,
    )


def _pr_legend_handles(tools: tuple[str, ...]) -> list[Line2D]:
    """Line2D handles matching plotted curves (assembler color × model linestyle)."""
    handles: list[Line2D] = []
    for tool in tools:
        pretty_asm = ASSEMBLER_DISPLAY.get(tool, tool.title())
        for model in ("baseline", "rf", "xgb"):
            color, ls, lw = _curve_plot_style(tool, model)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    label=f"{pretty_asm} {MODEL_LEGEND_LABELS[model]}",
                )
            )
    return handles


def _add_pr_figure_legend(fig: plt.Figure, tools: tuple[str, ...]) -> None:
    handles = _pr_legend_handles(tools)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, _LEGEND_FIG_Y),
        bbox_transform=fig.transFigure,
        fontsize=_LEGEND_FONTSIZE,
        ncol=_LEGEND_NCOL,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5,
        handlelength=2.4,
        borderaxespad=0.0,
    )


def _plot_assembler_curves_on_ax(
    ax: plt.Axes,
    *,
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] | None,
    assembler: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Draw one assembler's baseline/RF/XGB curves; return recall/precision arrays for limits."""
    recalls_all: list[np.ndarray] = []
    precisions_all: list[np.ndarray] = []
    if not curves:
        return recalls_all, precisions_all
    for model in ("baseline", "rf", "xgb"):
        if model not in curves:
            continue
        rec, pre, _ap = curves[model]
        recalls_all.append(rec)
        precisions_all.append(pre)
        color, ls, lw = _curve_plot_style(assembler, model)
        ax.plot(rec, pre, color=color, linewidth=lw, linestyle=ls)
    return recalls_all, precisions_all


def _finalize_pr_panel_ax(
    ax: plt.Axes,
    *,
    metric: str,
    recalls_all: list[np.ndarray],
    precisions_all: list[np.ndarray],
) -> None:
    _ = metric, precisions_all
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, _recall_xlim_upper(*recalls_all) if recalls_all else 1.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_box_aspect(1)


def _panel_label(idx: int) -> str:
    return f"({chr(ord('a') + idx)})"


def _plot_stacked_pr_figure(
    spec: StackedPrFigureSpec,
    curve_index: dict[CurveKey, dict[str, tuple[np.ndarray, np.ndarray, float]]],
    out_path: Path,
    *,
    subset: str,
) -> bool:
    n_panels = len(spec.datasets)
    if n_panels != spec.nrows * spec.ncols:
        raise ValueError(f"{spec.figure_id}: expected {spec.nrows}x{spec.ncols} panels, got {n_panels} datasets")

    fig, axes = plt.subplots(
        spec.nrows,
        spec.ncols,
        figsize=(spec.ncols * _SQUARE_PANEL_IN, spec.nrows * _SQUARE_PANEL_IN),
        dpi=_SQUARE_DPI,
        constrained_layout=False,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    missing = 0
    for idx, (ax, ds_key) in enumerate(zip(axes_flat, spec.datasets)):
        row = idx // spec.ncols
        col = idx % spec.ncols
        if spec.row_group_titles and col == 0:
            ax.set_ylabel(spec.row_group_titles[row], fontsize=11, labelpad=8)
        title = f"{_panel_label(idx)} {pretty_dataset_label(ds_key)}"
        ax.set_title(title, fontsize=10, pad=4)

        recalls_all: list[np.ndarray] = []
        precisions_all: list[np.ndarray] = []
        for tool in spec.tools:
            key: CurveKey = (subset, ds_key, tool, spec.metric)
            curves = curve_index.get(key)
            recs, pres = _plot_assembler_curves_on_ax(ax, curves=curves, assembler=tool)
            recalls_all.extend(recs)
            precisions_all.extend(pres)
            if not curves:
                missing += 1
        _finalize_pr_panel_ax(
            ax,
            metric=spec.metric,
            recalls_all=recalls_all,
            precisions_all=precisions_all,
        )

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=_GRID_BOTTOM,
        top=_GRID_TOP,
        hspace=_GRID_HSPACE,
        wspace=_GRID_WSPACE,
    )
    _add_pr_figure_legend(fig, spec.tools)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    if missing:
        print(f"[telos_repro] {spec.figure_id}: {missing} missing assembler curves (subset={subset})")
    return missing == 0


def _metric_spec_for_site(site: str):
    for spec in CORE_BENCHMARK_METRICS:
        if spec[0] == f"stage1_{site}":
            return spec
        elif spec[0] == f"{site}_pr":
            return spec
    raise ValueError(f"No benchmark metric spec for stage1_{site}")


# --- Jaccard similarity between assembler site sets (50 bp tolerance) ---


def _assembler_pair_for_dataset(v1_key: str) -> tuple[str, str]:
    """StringTie vs IsoQuant (long-read) or StringTie vs Scallop2 (short-read)."""
    if str(v1_key).startswith("SRR"):
        return ("stringtie", "scallop2")
    return ("stringtie", "isoquant")


def _parse_site_key(site: str) -> tuple[str, str, int, str]:
    site_type, chrom, pos, strand = site.split(":", 3)
    return site_type, chrom, int(pos), strand


def _parse_sites_for_jaccard(site_set: set[str]) -> tuple[dict[tuple[str, str, str], dict[int, list[str]]], dict]:
    parsed: dict[tuple[str, str, str], dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for site in site_set:
        site_type, chrom, pos, strand = _parse_site_key(site)
        parsed[(site_type, chrom, strand)][pos].append(site)
    return parsed, {}


def _find_site_intersection(set_a: set[str], set_b: set[str], *, max_dist: int = JACCARD_MAX_DIST_BP) -> set[str]:
    """Sites in *set_a* with a counterpart in *set_b* within *max_dist* bp (same type/chrom/strand)."""
    if not set_a or not set_b:
        return set()
    parsed_a, _ = _parse_sites_for_jaccard(set_a)
    parsed_b, _ = _parse_sites_for_jaccard(set_b)
    intersection: set[str] = set()
    for key in parsed_a:
        if key not in parsed_b:
            continue
        positions_a = sorted(parsed_a[key].keys())
        positions_b = sorted(parsed_b[key].keys())
        j = 0
        for pos_a in positions_a:
            while j < len(positions_b) and positions_b[j] < pos_a - max_dist:
                j += 1
            k = j
            while k < len(positions_b) and positions_b[k] <= pos_a + max_dist:
                for site_a in parsed_a[key][pos_a]:
                    intersection.add(site_a)
                break
    return intersection


def _jaccard_both_fraction(set_a: set[str], set_b: set[str], *, max_dist: int = JACCARD_MAX_DIST_BP) -> float:
    """
    Match v1 ``plot_venn.py`` ``both_pct``: |intersection| / (|only_a| + |only_b| + |both|).
    """
    if not set_a and not set_b:
        return float("nan")
    inter = _find_site_intersection(set_a, set_b, max_dist=max_dist)
    only_a = len(set_a - inter)
    only_b = len(set_b - inter)
    both = len(inter)
    total = only_a + only_b + both
    if total == 0:
        return float("nan")
    return both / total


def _site_id_set_from_scored(
    sites_df: pd.DataFrame,
    site_type: str,
    *,
    score_col: str | None = None,
    threshold: float = JACCARD_SCORE_THRESHOLD,
) -> set[str]:
    st = site_type.strip().upper()
    sub = sites_df[sites_df["site_type"].astype(str).str.upper() == st].copy()
    if sub.empty:
        return set()
    if score_col is not None:
        scores = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0)
        sub = sub[scores > threshold]
    out: set[str] = set()
    for _, r in sub.iterrows():
        st_low = str(r["site_type"]).strip().lower()
        chrom = str(r["chrom"]).strip()
        pos = int(pd.to_numeric(r["position"], errors="coerce"))
        strand = str(r["strand"]).strip()
        out.add(f"{st_low}:{chrom}:{pos}:{strand}")
    return out


def _read_sites_scored(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        dtype={"site_type": str, "chrom": str, "position": "Int64", "strand": str},
    )


def _compute_jaccard_table(
    root: Path,
    *,
    train_annotation: str,
    test_annotation: str,
) -> pd.DataFrame:
    """
    Per canonical dataset: Jaccard between StringTie and IsoQuant/Scallop2 site sets.

    Baseline uses all scored assembly sites; RF/XGB keep sites with p > 0.5.
    """
    sites_by_dataset: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    meta_by_dataset: dict[str, dict[str, str]] = {}

    for summary_csv in sorted(root.glob("*/reports/benchmark_summary.csv")):
        run_dir = summary_csv.parent.parent
        parsed = _parse_run_id(run_dir.name)
        if parsed is None:
            continue
        data_type, train_ann, test_ann, _variant = parsed
        if train_ann != train_annotation or test_ann != test_annotation:
            continue
        df = pd.read_csv(summary_csv)
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        for _, r in df.iterrows():
            test_id = str(r.get("test_id", "")).strip()
            if not test_id:
                continue
            asm = _assembler_from_test_id(test_id)
            if not asm:
                asm = str(r.get("assembler_id", "")).strip().lower()
            ds_key = v1_dataset_key(data_type, test_id)
            pred_out_raw = str(r.get("predict_outdir", "")).strip()
            pred_out = Path(pred_out_raw) if pred_out_raw else (run_dir / "tests" / test_id)
            sites_path = pred_out / "predictions" / "sites.scored.tsv"
            sites_df = _read_sites_scored(sites_path)
            if sites_df.empty:
                continue
            sites_by_dataset[ds_key][asm] = sites_df
            meta_by_dataset[ds_key] = {"data_type": data_type, "test_id": test_id}

    rows: list[dict[str, Any]] = []
    for ds_key in V1_CANONICAL_DATASET_ORDER:
        tool_a, tool_b = _assembler_pair_for_dataset(ds_key)
        asm_sites = sites_by_dataset.get(ds_key, {})
        if tool_a not in asm_sites or tool_b not in asm_sites:
            continue
        df_a = asm_sites[tool_a]
        df_b = asm_sites[tool_b]
        meta = meta_by_dataset.get(ds_key, {})
        data_type = meta.get("data_type", v1_group_for_datatype(ds_key))
        for site_type in ("tss", "tes"):
            base_a = _site_id_set_from_scored(df_a, site_type)
            base_b = _site_id_set_from_scored(df_b, site_type)
            rf_a = _site_id_set_from_scored(df_a, site_type, score_col="p_site_rf")
            rf_b = _site_id_set_from_scored(df_b, site_type, score_col="p_site_rf")
            xgb_a = _site_id_set_from_scored(df_a, site_type, score_col="p_site_xgb")
            xgb_b = _site_id_set_from_scored(df_b, site_type, score_col="p_site_xgb")
            rows.append(
                {
                    "v1_dataset_key": ds_key,
                    "data_type": data_type,
                    "test_id": meta.get("test_id", ds_key),
                    "tool_1": tool_a,
                    "tool_2": tool_b,
                    "site_type": site_type.upper(),
                    "baseline": _jaccard_both_fraction(base_a, base_b),
                    "rf": _jaccard_both_fraction(rf_a, rf_b),
                    "xgb": _jaccard_both_fraction(xgb_a, xgb_b),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(rows)
    out["v1_group"] = [v1_group_for_datatype(str(dt)) for dt in out["data_type"]]
    out["dataset_label"] = [pretty_dataset_label(k) for k in out["v1_dataset_key"]]
    ord_map = {k: i for i, k in enumerate(V1_CANONICAL_DATASET_ORDER)}
    out["_ord"] = out["v1_dataset_key"].map(lambda k: ord_map.get(k, 999))
    return out.sort_values(["site_type", "_ord"], kind="stable").drop(columns="_ord")


_JACCARD_CACHE_NAME = "jaccard_similarity_long.csv"
_JACCARD_CACHE_COLUMNS = frozenset(
    {"v1_dataset_key", "site_type", "baseline", "rf", "xgb", "dataset_label"}
)


def _load_jaccard_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return None
    if df.empty or not _JACCARD_CACHE_COLUMNS.issubset(df.columns):
        return None
    return df


def _order_jaccard_for_plot(jaccard_df: pd.DataFrame, *, site_type: str) -> pd.DataFrame:
    sub = jaccard_df[jaccard_df["site_type"].astype(str).str.upper() == site_type.upper()].copy()
    if sub.empty:
        return sub
    ord_map = {k: i for i, k in enumerate(V1_CANONICAL_DATASET_ORDER)}
    sub["_ord"] = sub["v1_dataset_key"].map(lambda k: ord_map.get(k, 999))
    return sub.sort_values("_ord", kind="stable").drop(columns="_ord")


def _plot_stacked_jaccard_bars(
    root: Path,
    outdir: Path,
    *,
    train_annotation: str,
    test_annotation: str,
    skip: bool = False,
    recompute: bool = False,
) -> None:
    """Stacked TSS/TES Jaccard bars (baseline vs Telos-RF vs Telos-XGB) per dataset."""
    if skip:
        print("[telos_repro] skip jaccard bars (--skip-jaccard)")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / _JACCARD_CACHE_NAME
    jaccard_df: pd.DataFrame | None = None
    if not recompute:
        jaccard_df = _load_jaccard_cache(cache_path)
        if jaccard_df is not None:
            print(f"[telos_repro] jaccard: using cache {cache_path}")

    if jaccard_df is None:
        print("[telos_repro] jaccard: computing site-set similarities (slow; cached afterward)")
        jaccard_df = _compute_jaccard_table(
            root,
            train_annotation=train_annotation,
            test_annotation=test_annotation,
        )
        if jaccard_df.empty:
            print("[telos_repro] skip jaccard bars: no site-set pairs found")
            return
        jaccard_df.to_csv(cache_path, index=False)
    elif jaccard_df.empty:
        print("[telos_repro] skip jaccard bars: empty cache")
        return

    tool_a, tool_b = _assembler_pair_for_dataset("pacbio_ENCFF694DIE")
    pair_label = f"{ASSEMBLER_DISPLAY.get(tool_a, tool_a.title())} vs {ASSEMBLER_DISPLAY.get(tool_b, tool_b.title())}"
    sr_a, sr_b = _assembler_pair_for_dataset("SRR307911")
    sr_label = (
        f"{ASSEMBLER_DISPLAY.get(sr_a, sr_a.title())} vs "
        f"{ASSEMBLER_DISPLAY.get(sr_b, sr_b.title())} (SR)"
    )

    panels: list[BarPanel] = []
    for site_type, row_label in (("TSS", "TSS"), ("TES", "TES")):
        ordered = _order_jaccard_for_plot(jaccard_df, site_type=site_type)
        if ordered.empty:
            continue
        panels.append(
            BarPanel(
                row_label=row_label,
                labels=ordered["dataset_label"].astype(str).tolist(),
                base_v=ordered["baseline"].to_numpy(dtype=float),
                rf_v=ordered["rf"].to_numpy(dtype=float),
                xgb_v=ordered["xgb"].to_numpy(dtype=float),
                ordered=ordered,
            )
        )

    if not panels:
        print("[telos_repro] skip jaccard bars: no TSS/TES rows")
        return

    fig_path = outdir / "fig13_jaccard_similarity_bars.pdf"
    draw_stacked_bar_panels(
        panels,
        fig_path,
        show_aupr_ylabel=False,
        figure_ylabel="Jaccard\nsimilarity",
    )
    print(f"[telos_repro] fig13 jaccard bars -> {fig_path} ({pair_label}; {sr_label} on SR)")


def _plot_stacked_aupr_bars(
    root: Path,
    outdir: Path,
    *,
    train_annotation: str,
    test_annotation: str,
) -> None:
    """Figures 7–8: stacked AuPR bars (StringTie panel + IsoQuant/Scallop2 panel).
    Figures 12: stacked AuPR bars (transcript PR).
    """
    df = load_benchmark_summaries(
        root,
        row_filter=lambda d: d[
            (d["train_annotation"].astype(str) == train_annotation)
            & (d["test_annotation"].astype(str) == test_annotation)
        ],
    )
    for site, fig_name in (("tss", "fig07_stage1_tss_aupr_bars.pdf"), ("tes", "fig08_stage1_tes_aupr_bars.pdf"), ("transcript", "fig12_transcript_pr_aupr_bars.pdf")):
        spec = _metric_spec_for_site(site)
        per_row = _extract_metric_scores(df, spec)
        if per_row.empty:
            print(f"[telos_repro] skip {fig_name}: no {site} scores")
            continue
        panels: list[BarPanel] = []
        for panel in CROSS_ANNOTATION_TOOL_PANELS:
            ordered = order_scores_for_v1_tool_panel(per_row, panel=panel)
            if ordered.empty:
                continue
            labels = ordered["dataset_label"].astype(str).tolist()
            panels.append(
                BarPanel(
                    row_label="StringTie" if panel == "stringtie" else "IsoQuant / Scallop2",
                    labels=labels,
                    base_v=ordered["baseline"].to_numpy(dtype=float),
                    rf_v=ordered["rf"].to_numpy(dtype=float),
                    xgb_v=ordered["xgb"].to_numpy(dtype=float),
                    ordered=annotate_v1_labels(ordered),
                )
            )
        if panels:
            draw_stacked_bar_panels(panels, outdir / fig_name)


def _plot_legacy_per_dataset(
    rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    """Legacy 1×3 TSS/TES/transcript PNGs per test (optional debug output)."""
    by_key: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]]] = {}
    for r in rows:
        k = f"{r['run_id']}__{r['test_id']}"
        by_key.setdefault(k, {}).setdefault(r["metric"], {})[r["model"]] = (r["recall"], r["precision"], r["ap"])
    for key, curves in by_key.items():
        _plot_dataset_curves(dataset_label=key, curves=curves, out_png=outdir / f"{key.replace('/', '_')}.png")


def _plot_dataset_curves(
    *,
    dataset_label: str,
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)
    for ax, metric in zip(axes, ("tss", "tes", "transcript")):
        ax.set_title(metric.upper())
        recalls_all: list[np.ndarray] = []
        precisions_all: list[np.ndarray] = []
        for tool in ("stringtie", "isoquant", "scallop2"):
            key_curves = curves.get(metric)
            recs, pres = _plot_assembler_curves_on_ax(ax, curves=key_curves, assembler=tool)
            recalls_all.extend(recs)
            precisions_all.extend(pres)
        _finalize_pr_panel_ax(
            ax, metric=metric, recalls_all=recalls_all, precisions_all=precisions_all
        )
    fig.suptitle(dataset_label, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _interp_precision_on_grid(recall: np.ndarray, precision: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    r_u, idx = np.unique(r, return_index=True)
    p_u = p[idx]
    return np.interp(grid, r_u, p_u, left=p_u[0], right=p_u[-1])


def _plot_combo_merged(
    *,
    combo_label: str,
    curve_rows: list[dict[str, Any]],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=170)
    grid = np.linspace(0.0, 1.0, 201)
    style = {"rf": "#1f77b4", "xgb": "#ff7f0e", "baseline": "#444444"}
    for ax, metric in zip(axes, ("tss", "tes", "transcript")):
        ax.set_title(f"{metric.upper()} merged")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        merged_recalls: list[np.ndarray] = []
        for model in ("rf", "xgb", "baseline"):
            picks = [r for r in curve_rows if r["metric"] == metric and r["model"] == model]
            for r in picks:
                merged_recalls.append(r["recall"])
        ax.set_xlim(0.0, _recall_xlim_upper(*merged_recalls) if merged_recalls else 1.0)
        for model in ("rf", "xgb", "baseline"):
            picks = [r for r in curve_rows if r["metric"] == metric and r["model"] == model]
            if not picks:
                continue
            mats = np.vstack(
                [_interp_precision_on_grid(r["recall"], r["precision"], grid) for r in picks]
            )
            mean_p = mats.mean(axis=0)
            std_p = mats.std(axis=0)
            color = style[model]
            ls = "-" if model != "baseline" else "--"
            hi = np.clip(mean_p + std_p, 0, 1)
            lo = np.clip(mean_p - std_p, 0, 1)
            ax.plot(grid, mean_p, color=color, linewidth=2.5, linestyle=ls, label=f"{model.upper()} (n={len(picks)})")
            ax.fill_between(grid, lo, hi, color=color, alpha=0.14)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle(combo_label, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def run(
    root: Path,
    outdir: Path,
    subset: str,
    *,
    train_annotation: str,
    test_annotation: str,
    write_legacy: bool,
    write_publication: bool,
    skip_jaccard: bool = False,
    recompute_jaccard: bool = False,
) -> int:
    apply_plot_style()
    pairing_dir = f"{train_annotation}__to__{test_annotation}"
    pub_out = outdir / pairing_dir

    curve_index = _load_curve_index(
        root,
        train_annotation=train_annotation,
        test_annotation=test_annotation,
        subset=subset if subset != "both" else "all",
    )
    if not curve_index:
        raise ValueError(
            f"No PR curves under {root} for train={train_annotation} test={test_annotation}."
        )

    rows: list[dict[str, Any]] = []
    for (subset_name, ds_key, asm, metric), model_map in curve_index.items():
        for model, (rec, pre, ap) in model_map.items():
            rows.append(
                {
                    "subset": subset_name,
                    "v1_dataset_key": ds_key,
                    "assembler_id": asm,
                    "metric": metric,
                    "model": model,
                    "ap": ap,
                    "recall": rec,
                    "precision": pre,
                }
            )

    ap_df = pd.DataFrame(
        [
            {
                "subset": r["subset"],
                "v1_dataset_key": r["v1_dataset_key"],
                "assembler_id": r["assembler_id"],
                "metric": r["metric"],
                "model": r["model"],
                "ap": r["ap"],
            }
            for r in rows
        ]
    )
    pub_out.mkdir(parents=True, exist_ok=True)
    ap_df.to_csv(pub_out / "pr_ap_long.csv", index=False)

    # convert ap_df where for each model we have a column with the ap value to a wide format dataframe 
    ap_df_wide = ap_df.pivot(index=["subset", "v1_dataset_key", "assembler_id", "metric"], columns="model", values="ap")
    ap_df_wide["pct_gain_rf"] = (ap_df_wide["rf"] - ap_df_wide["baseline"]) / ap_df_wide["baseline"]
    ap_df_wide["pct_gain_xgb"] = (ap_df_wide["xgb"] - ap_df_wide["baseline"]) / ap_df_wide["baseline"]
    ap_df_wide.reset_index(inplace=True)
    ap_df_wide.drop(columns=["subset","baseline", "rf", "xgb"], inplace=True)
    ap_df_wide.to_csv(pub_out / "pr_ap_wide.csv",float_format='%.2f')

    # print(ap_df_wide.columns)
    print(ap_df_wide[ap_df_wide["metric"] == "tss"][["pct_gain_rf", "pct_gain_xgb"]].describe().to_string())
    print(ap_df_wide[ap_df_wide["metric"] == "tes"][["pct_gain_rf", "pct_gain_xgb"]].describe().to_string())
    print(ap_df_wide[ap_df_wide["metric"] == "transcript"][["pct_gain_rf", "pct_gain_xgb"]].describe().to_string())

    active_subset = subset if subset != "both" else "all"
    if write_publication:
        for spec in PUBLICATION_PR_FIGURES:
            out_pdf = pub_out / spec.filename
            ok = _plot_stacked_pr_figure(spec, curve_index, out_pdf, subset=active_subset)
            status = "ok" if ok else "partial"
            print(f"[telos_repro] {spec.figure_id} -> {out_pdf} ({status})")
        _plot_stacked_aupr_bars(
            root,
            pub_out,
            train_annotation=train_annotation,
            test_annotation=test_annotation,
        )
        print(f"[telos_repro] fig07/fig08 AuPR bars -> {pub_out}")
        _plot_stacked_jaccard_bars(
            root,
            pub_out,
            train_annotation=train_annotation,
            test_annotation=test_annotation,
            skip=skip_jaccard,
            recompute=recompute_jaccard,
        )

    if write_legacy:
        legacy = pub_out / "legacy"
        _plot_legacy_per_dataset(rows, legacy / "per_dataset")
        combo_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in rows:
            combo_groups.setdefault((r["subset"], r["metric"], r["assembler_id"]), []).append(r)
        for (subset_name, metric, asm), grp in combo_groups.items():
            _plot_combo_merged(
                combo_label=f"subset={subset_name} | metric={metric} | assembler={asm}",
                curve_rows=grp,
                out_png=legacy / "merged" / f"{subset_name}__{metric}__{asm}.png",
            )

    print(f"[telos_repro] PR outputs under: {pub_out.resolve()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Stacked PR-curve figures from benchmark predictions.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("runs/cross_annotation_repro"),
        help="Root directory containing benchmark run folders",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("plot_v2/cross_annotation_repro"),
        help="Output root (figures go under <outdir>/<train>__to__<test>/)",
    )
    p.add_argument(
        "--train-annotation",
        default="gencode",
        help="Filter runs to this train annotation (default: gencode)",
    )
    p.add_argument(
        "--test-annotation",
        default="gencode",
        help="Filter runs to this test annotation (default: gencode)",
    )
    p.add_argument(
        "--subset",
        choices=("all", "novel", "both"),
        default="all",
        help="Curve subset: all sites, novel-only, or both (publication figures use all unless novel)",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Also write legacy per-dataset PNGs and merged curves under legacy/",
    )
    p.add_argument(
        "--no-publication",
        action="store_true",
        help="Skip stacked publication PDFs (fig 3–8, 10–12)",
    )
    p.add_argument(
        "--skip-jaccard",
        action="store_true",
        help="Skip fig13 Jaccard bars (site-set comparison is slow to compute)",
    )
    p.add_argument(
        "--recompute-jaccard",
        action="store_true",
        help="Recompute Jaccard table even if jaccard_similarity_long.csv exists",
    )
    args = p.parse_args()
    return run(
        args.root.resolve(),
        args.outdir.resolve(),
        args.subset,
        train_annotation=args.train_annotation.strip(),
        test_annotation=args.test_annotation.strip(),
        write_legacy=args.legacy,
        write_publication=not args.no_publication,
        skip_jaccard=args.skip_jaccard,
        recompute_jaccard=args.recompute_jaccard,
    )


if __name__ == "__main__":
    raise SystemExit(main())
