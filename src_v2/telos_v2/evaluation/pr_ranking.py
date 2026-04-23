"""
Precision–recall style curves from aligned score arrays (sklearn).

For **benchmark** transcript-level evaluation that re-runs gffcompare on score-injected GTFs, use
``telos_v2.evaluation.transcript_pr_pipeline`` instead: it runs
``gtfformat filter-chrom`` (validation split) → ``update-transcript-cov`` → ``gffcompare`` → ``gtfcuff roc`` / ``auc`` so the
ranking axis matches reference re-annotation.

The helpers here remain useful for synthetic tests or quick merges of a fixed tmap with
tabular scores; they do not substitute transcript coverage in the assembly GTF.

**Naming:** functions in this module return flat dicts with ``pr_*`` keys for historical CSV consumers.
The gffcompare-based pipeline (``run_transcript_pr_benchmark`` in ``transcript_pr_pipeline``)
uses ``transcript_pr_*`` keys when attaching results to benchmark rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from telos_v2.labels.transcript_labels import load_tmap_labels


@dataclass(frozen=True)
class PRCurve:
    """
    Precision–recall curve from :func:`sklearn.metrics.precision_recall_curve` plus AUPR.

    ``thresholds`` has length ``len(precision)-1`` in sklearn convention; consumers plotting or
    exporting should pad or align as in :func:`write_pr_comparison_outputs`.
    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    aupr: float


@dataclass(frozen=True)
class PRComparisonResult:
    """
    Side-by-side PR evaluation for a model score vector and a baseline (e.g. abundance) on identical rows.

    ``y_true``, ``scores_model``, and ``scores_baseline`` are already filtered to finite, aligned rows.
    """

    y_true: np.ndarray
    scores_model: np.ndarray
    scores_baseline: np.ndarray
    model: PRCurve
    baseline: PRCurve
    n_items: int
    n_positive: int
    abundance_column: str


def _pr_curve_from_scores(y_true: np.ndarray, scores: np.ndarray) -> PRCurve:
    """
    Build :class:`PRCurve` from binary ``y_true`` and real ``scores`` (higher = more confident positive).

    Drops non-finite scores. Raises if no finite rows remain.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        raise ValueError("No finite scores for PR curve")
    prec, rec, thr = precision_recall_curve(y, s)
    aupr = float(average_precision_score(y, s))
    return PRCurve(precision=prec, recall=rec, thresholds=thr, aupr=aupr)


def compute_pr_curves_vs_baseline(
    y_true: np.ndarray,
    scores_model: np.ndarray,
    scores_baseline: np.ndarray,
    *,
    abundance_column: str = "coverage",
) -> PRComparisonResult:
    """
    Build PR curves for model scores and baseline scores on the **same rows**.

    Rows with NaN/inf in either score are dropped (aligned mask). Requires at least one positive label
    in ``y_true`` after filtering so recall is defined.
    """
    y = np.asarray(y_true, dtype=int)
    m = np.asarray(scores_model, dtype=float)
    b = np.asarray(scores_baseline, dtype=float)
    if not (y.shape == m.shape == b.shape):
        raise ValueError("y_true, scores_model, scores_baseline must have the same shape")
    ok = np.isfinite(m) & np.isfinite(b)
    y = y[ok]
    m = m[ok]
    b = b[ok]
    if y.size == 0:
        raise ValueError("No rows with finite model and baseline scores")

    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive labels (cannot define recall)")
    return PRComparisonResult(
        y_true=y,
        scores_model=m,
        scores_baseline=b,
        model=_pr_curve_from_scores(y, m),
        baseline=_pr_curve_from_scores(y, b),
        n_items=int(y.size),
        n_positive=n_pos,
        abundance_column=abundance_column,
    )


def transcript_pr_vs_baseline_from_files(
    ranked_tsv: Path,
    cov_tsv: Path,
    tmap_path: Path,
    *,
    abundance_column: str = "coverage",
    pred_column: str = "pred_prob",
) -> PRComparisonResult:
    """
    Transcript-level PR: merge ranked transcript TSV, cov table, and gffcompare tmap labels.

    - Labels: ``class_code == '='`` → positive (same as ``load_tmap_labels``).
    - Model score: ``pred_prob`` (higher = better predicted transcript).
    - Baseline score: ``abundance_column`` (default ``coverage``), higher = stronger abundance signal.

    Dedupes ``ranked`` and ``cov`` by ``transcript_id`` keeping highest score / abundance per id.
    """
    ranked = pd.read_csv(ranked_tsv, sep="\t")
    if pred_column not in ranked.columns:
        raise ValueError(f"ranked TSV missing column {pred_column!r}: {ranked_tsv}")
    ranked = ranked.sort_values(pred_column, ascending=False).drop_duplicates(
        "transcript_id", keep="first"
    )
    cov = pd.read_csv(
        cov_tsv,
        sep="\t",
        dtype={"transcript_id": str},
    )
    if abundance_column not in cov.columns:
        raise ValueError(f"coverage TSV missing column {abundance_column!r}: {cov_tsv}")
    cov = cov.sort_values(abundance_column, ascending=False).drop_duplicates(
        "transcript_id", keep="first"
    )

    labels = load_tmap_labels(tmap_path)
    df = ranked[["transcript_id", pred_column]].merge(
        cov[["transcript_id", abundance_column]],
        on="transcript_id",
        how="inner",
    )
    df = df.merge(labels, on="transcript_id", how="inner")
    if df.empty:
        raise ValueError("Empty merge between ranked, cov, and tmap (check transcript_id overlap)")

    y = df["label"].to_numpy(dtype=int)
    m = df[pred_column].to_numpy(dtype=float)
    b = df[abundance_column].to_numpy(dtype=float)

    return compute_pr_curves_vs_baseline(
        y,
        m,
        b,
        abundance_column=abundance_column,
    )


def run_transcript_pr_benchmark_artifacts(
    ranked_tsv: Path,
    cov_tsv: Path,
    tmap_path: Path,
    reports_pr_dir: Path,
    *,
    abundance_column: str = "coverage",
    pred_column: str = "pred_prob",
    prefix: str = "transcript_pr",
    plot: bool = True,
    plot_filename: str = "transcript_pr.png",
) -> dict[str, Any]:
    """
    Compute transcript PR (model vs abundance baseline), write curve TSVs, metrics CSV, optional plot.

    Wraps :func:`transcript_pr_vs_baseline_from_files`, writes TSV/CSV under ``reports_pr_dir`` with
    ``prefix``, optionally saves matplotlib plot. Returns flat ``pr_*`` keys for benchmark row merge.
    """
    result = transcript_pr_vs_baseline_from_files(
        ranked_tsv,
        cov_tsv,
        tmap_path,
        abundance_column=abundance_column,
        pred_column=pred_column,
    )
    reports_pr_dir.mkdir(parents=True, exist_ok=True)
    m_path, b_path, summary_path = write_pr_comparison_outputs(reports_pr_dir, result, prefix=prefix)
    out: dict[str, Any] = {
        "pr_aupr_model": result.model.aupr,
        "pr_aupr_baseline": result.baseline.aupr,
        "pr_n_items": result.n_items,
        "pr_n_positive": result.n_positive,
        "pr_abundance_column": result.abundance_column,
        "pr_model_curve_tsv": str(m_path),
        "pr_baseline_curve_tsv": str(b_path),
        "pr_aupr_summary_csv": str(summary_path),
        "pr_plot_png": "",
    }
    if plot:
        try:
            plot_path = reports_pr_dir / plot_filename
            plot_pr_comparison(
                result,
                plot_path,
                title="Transcript PR: model vs abundance baseline",
            )
            out["pr_plot_png"] = str(plot_path)
        except ImportError:
            pass
    return out


def write_pr_comparison_outputs(
    out_dir: Path,
    result: PRComparisonResult,
    *,
    prefix: str = "pr",
) -> tuple[Path, Path, Path]:
    """
    Persist model and baseline PR curves as tab-separated ``precision/recall/threshold`` tables.

    Threshold column is padded to match sklearn curve length (first row NaN). Writes two-row summary CSV
    with AUPR and counts. Returns ``(model_tsv, baseline_tsv, summary_csv)`` paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / f"{prefix}_model_pr.tsv"
    b_path = out_dir / f"{prefix}_baseline_pr.tsv"

    def _write_curve_tsv(path: Path, curve: PRCurve) -> None:
        prec = curve.precision
        rec = curve.recall
        thr = curve.thresholds
        n = len(prec)
        thr_col = np.full(n, np.nan, dtype=float)
        if len(thr) == n - 1:
            thr_col[1:] = thr
        elif len(thr) == 0 and n <= 1:
            pass
        else:
            m = min(len(thr), n)
            thr_col[1 : 1 + m] = thr[:m]
        pd.DataFrame(
            {"precision": prec, "recall": rec, "threshold": thr_col},
        ).to_csv(path, sep="\t", index=False)

    _write_curve_tsv(m_path, result.model)
    _write_curve_tsv(b_path, result.baseline)
    summary = out_dir / f"{prefix}_aupr_summary.csv"
    pd.DataFrame(
        [
            {
                "curve": "model",
                "aupr": result.model.aupr,
                "n_items": result.n_items,
                "n_positive": result.n_positive,
                "abundance_column": result.abundance_column,
            },
            {
                "curve": "baseline",
                "aupr": result.baseline.aupr,
                "n_items": result.n_items,
                "n_positive": result.n_positive,
                "abundance_column": result.abundance_column,
            },
        ]
    ).to_csv(summary, index=False)
    return m_path, b_path, summary


def plot_pr_comparison(
    result: PRComparisonResult,
    out_path: Path,
    *,
    title: str = "Transcript precision–recall: model vs abundance baseline",
    dpi: int = 120,
) -> Path:
    """
    Draw model vs baseline precision–recall with AUPR in legend; axis limits [0,1] × [0,1.05].

    Raises:
        ImportError: If matplotlib is not installed.

    Returns:
        Resolved ``out_path`` written as PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_pr_comparison requires matplotlib") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    ax.plot(
        result.model.recall,
        result.model.precision,
        label=f"model (AUPR={result.model.aupr:.4f})",
        color="#1f77b4",
        linewidth=1.8,
    )
    ax.plot(
        result.baseline.recall,
        result.baseline.precision,
        label=f"baseline {result.abundance_column} (AUPR={result.baseline.aupr:.4f})",
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def site_pr_vs_baseline(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    prob_col: str = "p_site",
    abundance_col: str = "coverage",
) -> PRComparisonResult:
    """
    Site-level (TSS/TES) PR when ``df`` has aligned columns for labels, model prob, and abundance.

    Expects one row per site with binary ``label_col``, Stage I probability ``prob_col``,
    and raw abundance ``abundance_col`` (e.g. local coverage from the feature / cov merge).
    """
    need = {label_col, prob_col, abundance_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"site_pr_vs_baseline missing columns: {sorted(missing)}")
    sub = df[[label_col, prob_col, abundance_col]].dropna()
    y = sub[label_col].to_numpy(dtype=int)
    m = sub[prob_col].to_numpy(dtype=float)
    b = sub[abundance_col].to_numpy(dtype=float)
    return compute_pr_curves_vs_baseline(
        y,
        m,
        b,
        abundance_column=abundance_col,
    )
