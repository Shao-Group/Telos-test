"""
Plot human GENCODE feature-window sweep results: metrics vs window_size.

Reads ``benchmark_summary.csv`` from run folders under ``--root`` produced by
``human_gencode_feature_window_sweep.py``::

    <root>/
      sr__ws50__train_gencode__test_gencode/reports/benchmark_summary.csv
      sr__ws100__train_gencode__test_gencode/reports/benchmark_summary.csv
      ...

Usage::

  PYTHONPATH=src_v2 python src_v2/experiments/plot_human_gencode_feature_window_results.py \\
    --root runs/human_gencode_feature_window
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW_SWEEP_RE = re.compile(
    r"^(?P<data_type>.+?)__ws(?P<window_size>\d+)__train_gencode__test_gencode$"
)

METRIC_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "stage1_tss",
        "Stage I TSS (test AUPR)",
        "stage1_test_aupr_tss_rf",
        "stage1_test_aupr_tss_xgb",
        "stage1_test_aupr_tss_baseline",
    ),
    (
        "stage1_tes",
        "Stage I TES (test AUPR)",
        "stage1_test_aupr_tes_rf",
        "stage1_test_aupr_tes_xgb",
        "stage1_test_aupr_tes_baseline",
    ),
    (
        "stage2_tmap",
        "Stage II vs bundle tmap (AUPR)",
        "stage2_test_aupr_rf",
        "stage2_test_aupr_xgb",
        "stage2_test_aupr_baseline",
    ),
    (
        "transcript_pr",
        "Transcript PR (AUC vs ref)",
        "transcript_pr_auc_model_rf",
        "transcript_pr_auc_model_xgb",
        "transcript_pr_auc_baseline",
    ),
)


def _parse_run_id(run_id: str) -> tuple[str, int] | None:
    m = WINDOW_SWEEP_RE.match(run_id.strip())
    if not m:
        return None
    return m.group("data_type").strip(), int(m.group("window_size"))


def load_summaries(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for summary in sorted(root.glob("*/reports/benchmark_summary.csv")):
        run_id = summary.parent.parent.name
        parsed = _parse_run_id(run_id)
        if parsed is None:
            continue
        dt, ws = parsed
        df = pd.read_csv(summary)
        if df.empty:
            continue
        df["run_id"] = run_id
        df["data_type"] = dt
        df["window_size"] = ws
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No benchmark_summary.csv under {root} "
            "(expected folders like <dt>__ws<N>__train_gencode__test_gencode)."
        )
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["window_size"] = pd.to_numeric(out["window_size"], errors="coerce")
    return out


def _mean_by_window(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    sub = df[["data_type", "window_size", value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    return (
        sub.dropna(subset=[value_col])
        .groupby(["data_type", "window_size"], as_index=False)[value_col]
        .mean()
        .sort_values(["data_type", "window_size"])
    )


def plot_metric_lines(df: pd.DataFrame, *, outdir: Path, stem: str, title: str, rf_col: str, xgb_col: str, bl_col: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    data_types = sorted(df["data_type"].dropna().unique())
    n_dt = len(data_types)
    if n_dt == 0:
        return

    fig, axes = plt.subplots(1, n_dt, figsize=(4.2 * n_dt, 4.0), squeeze=False)
    for ax, dt in zip(axes[0], data_types, strict=True):
        sub = df[df["data_type"] == dt]
        for label, col, color, marker in (
            ("RF", rf_col, "#1f77b4", "o"),
            ("XGB", xgb_col, "#ff7f0e", "s"),
            ("Baseline", bl_col, "#7f7f7f", "^"),
        ):
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
                label=label,
                linewidth=2,
                markersize=7,
            )
        ax.set_title(dt)
        ax.set_xlabel("feature window_size (bp)")
        ax.set_ylabel("mean score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_plotting(*, root: Path, plot_dir: Path | None) -> Path:
    from telos_v2.plotting.feature_window_plots import run_window_plots

    root = root.resolve()
    plot_dir = (plot_dir if plot_dir is not None else root / "plots").resolve()
    run_window_plots(root, plot_dir)
    print(f"[plot] figures under {plot_dir}")
    return plot_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Plot GENCODE feature-window sweep benchmark summaries.")
    p.add_argument("--root", type=Path, default=Path("runs/human_gencode_feature_window"))
    p.add_argument("--plot-dir", type=Path, default=None, help="Output directory (default: <root>/plots).")
    args = p.parse_args()
    try:
        run_plotting(root=args.root, plot_dir=args.plot_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
