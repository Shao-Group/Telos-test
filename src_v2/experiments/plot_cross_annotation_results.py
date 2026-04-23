"""
Plot cross-annotation benchmark results (non-bubble views).

Usage:
  conda run -n irtesam-berth python src_v2/experiments/plot_cross_annotation_results.py
"""

from __future__ import annotations

from pathlib import Path
import re
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("runs/cross_annotation_repro")
OUTDIR = ROOT / "reports" / "plots"

RUN_RE = re.compile(
    r"^(?P<data_type>[^_]+)__train_(?P<train>[^_]+)__test_(?P<test>[^_]+)"
    r"(?:__(?P<variant>.+))?$"
)


def _parse_run_id(run_id: str) -> tuple[str, str, str, str] | None:
    m = RUN_RE.match(run_id.strip())
    if not m:
        return None
    variant = (m.group("variant") or "").strip()
    return m.group("data_type"), m.group("train"), m.group("test"), variant


def load_all_summaries(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for summary in sorted(root.glob("*/reports/benchmark_summary.csv")):
        run_id = summary.parent.parent.name
        parsed = _parse_run_id(run_id)
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
        raise FileNotFoundError(f"No benchmark_summary.csv files found under {root}")
    return pd.concat(rows, axis=0, ignore_index=True)


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    metric_specs = [
        ("stage1_tss", "stage1_test_aupr_tss_rf", "stage1_test_aupr_tss_xgb", "stage1_test_aupr_tss_baseline"),
        ("stage1_tes", "stage1_test_aupr_tes_rf", "stage1_test_aupr_tes_xgb", "stage1_test_aupr_tes_baseline"),
        (
            "transcript",
            "transcript_pr_auc_model_rf",
            "transcript_pr_auc_model_xgb",
            _pick_col(df, "transcript_pr_auc_baseline", "transcript_pr_auc_baseline_rf"),
        ),
        (
            "stage1_tss_novel",
            "stage1_test_aupr_tss_novel_rf",
            "stage1_test_aupr_tss_novel_xgb",
            "stage1_test_aupr_tss_novel_baseline",
        ),
        (
            "stage1_tes_novel",
            "stage1_test_aupr_tes_novel_rf",
            "stage1_test_aupr_tes_novel_xgb",
            "stage1_test_aupr_tes_novel_baseline",
        ),
        (
            "transcript_novel",
            "stage2_test_aupr_novel_rf",
            "stage2_test_aupr_novel_xgb",
            "stage2_test_aupr_novel_baseline",
        ),
    ]

    for _, r in df.iterrows():
        asm = str(r.get("assembler_id", "")).strip().lower() or "unknown"
        for group, rf_col, xgb_col, base_col in metric_specs:
            if not base_col or base_col not in df.columns or rf_col not in df.columns or xgb_col not in df.columns:
                continue
            base = pd.to_numeric(pd.Series([r.get(base_col)]), errors="coerce").iloc[0]
            rf = pd.to_numeric(pd.Series([r.get(rf_col)]), errors="coerce").iloc[0]
            xgb = pd.to_numeric(pd.Series([r.get(xgb_col)]), errors="coerce").iloc[0]
            if pd.isna(base):
                continue
            for model, val in (("rf", rf), ("xgb", xgb)):
                if pd.isna(val):
                    continue
                delta = float(val - base)
                rel = float(delta / (abs(base) + 1e-12))
                out_rows.append(
                    {
                        "run_id": r["run_id"],
                        "test_id": r.get("test_id", ""),
                        "data_type": r["data_type"],
                        "train_annotation": r["train_annotation"],
                        "test_annotation": r["test_annotation"],
                        "run_variant": r.get("run_variant", ""),
                        "assembler_id": asm,
                        "metric_group": group,
                        "model": model,
                        "model_value": float(val),
                        "baseline_value": float(base),
                        "delta_vs_baseline": delta,
                        "relative_gain": rel,
                        "relative_gain_pct": rel * 100.0,
                        "win_vs_baseline": bool(delta > 0.0),
                    }
                )
    if not out_rows:
        raise ValueError("No comparable model/baseline metric rows found.")
    return pd.DataFrame(out_rows)


def _heatmap_winrate(cmp_df: pd.DataFrame, metric_group: str, model: str, out_path: Path) -> None:
    sub = cmp_df[(cmp_df["metric_group"] == metric_group) & (cmp_df["model"] == model)].copy()
    if sub.empty:
        return
    dtypes = sorted(sub["data_type"].unique())
    assemblers = sorted(sub["assembler_id"].unique())
    fig, axes = plt.subplots(
        nrows=len(assemblers), ncols=len(dtypes), figsize=(3.8 * len(dtypes), 3.2 * len(assemblers))
    )
    axes = np.atleast_2d(axes)

    for i, asm in enumerate(assemblers):
        for j, dt in enumerate(dtypes):
            ax = axes[i, j]
            cell = sub[(sub["assembler_id"] == asm) & (sub["data_type"] == dt)]
            if cell.empty:
                ax.axis("off")
                continue
            piv = (
                cell.groupby(["train_annotation", "test_annotation"])["win_vs_baseline"]
                .mean()
                .unstack(fill_value=np.nan)
            )
            train_labels = list(piv.index)
            test_labels = list(piv.columns)
            m = piv.to_numpy(dtype=float)
            im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap="RdYlGn")
            ax.set_xticks(range(len(test_labels)), test_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(train_labels)), train_labels)
            ax.set_title(f"{dt} | {asm}")
            for rr in range(m.shape[0]):
                for cc in range(m.shape[1]):
                    if np.isfinite(m[rr, cc]):
                        ax.text(cc, rr, f"{100.0*m[rr, cc]:.0f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Win-rate vs baseline")
    fig.suptitle(f"{metric_group}: {model.upper()} win-rate vs baseline", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _delta_boxplot(cmp_df: pd.DataFrame, metric_group: str, out_path: Path) -> None:
    sub = cmp_df[cmp_df["metric_group"] == metric_group].copy()
    if sub.empty:
        return
    dtypes = sorted(sub["data_type"].unique())
    assemblers = sorted(sub["assembler_id"].unique())
    fig, axes = plt.subplots(
        nrows=len(assemblers), ncols=1, figsize=(4.0 * len(dtypes), 3.6 * len(assemblers)), squeeze=False
    )
    for i, asm in enumerate(assemblers):
        ax = axes[i, 0]
        data_asm = sub[sub["assembler_id"] == asm]
        positions: list[float] = []
        values: list[np.ndarray] = []
        colors: list[str] = []
        labels: list[str] = []
        x = 1.0
        for dt in dtypes:
            for model, color in (("rf", "#1f77b4"), ("xgb", "#ff7f0e")):
                arr = data_asm[(data_asm["data_type"] == dt) & (data_asm["model"] == model)][
                    "delta_vs_baseline"
                ].dropna()
                if arr.empty:
                    continue
                positions.append(x)
                values.append(arr.to_numpy())
                colors.append(color)
                labels.append(f"{dt}\n{model}")
                x += 1.0
            x += 0.4
        if not values:
            ax.axis("off")
            continue
        bp = ax.boxplot(values, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Model - baseline")
        ax.set_title(f"{metric_group} delta distribution | assembler={asm}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _pct_gain_heatmap(cmp_df: pd.DataFrame, metric_group: str, model: str, out_path: Path) -> None:
    sub = cmp_df[(cmp_df["metric_group"] == metric_group) & (cmp_df["model"] == model)].copy()
    if sub.empty:
        return
    dtypes = sorted(sub["data_type"].unique())
    assemblers = sorted(sub["assembler_id"].unique())
    fig, axes = plt.subplots(
        nrows=len(assemblers), ncols=len(dtypes), figsize=(3.8 * len(dtypes), 3.2 * len(assemblers))
    )
    axes = np.atleast_2d(axes)
    vmax = max(1.0, float(sub["relative_gain_pct"].abs().quantile(0.95)))
    vmin = -vmax
    for i, asm in enumerate(assemblers):
        for j, dt in enumerate(dtypes):
            ax = axes[i, j]
            cell = sub[(sub["assembler_id"] == asm) & (sub["data_type"] == dt)]
            if cell.empty:
                ax.axis("off")
                continue
            piv = (
                cell.groupby(["train_annotation", "test_annotation"])["relative_gain_pct"]
                .median()
                .unstack(fill_value=np.nan)
            )
            train_labels = list(piv.index)
            test_labels = list(piv.columns)
            m = piv.to_numpy(dtype=float)
            im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_xticks(range(len(test_labels)), test_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(train_labels)), train_labels)
            ax.set_title(f"{dt} | {asm}")
            for rr in range(m.shape[0]):
                for cc in range(m.shape[1]):
                    if np.isfinite(m[rr, cc]):
                        ax.text(cc, rr, f"{m[rr, cc]:.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Median % gain vs baseline")
    fig.suptitle(f"{metric_group}: {model.upper()} median % gain vs baseline", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _transcript_ranked_scatter(cmp_df: pd.DataFrame, out_path: Path) -> None:
    sub = cmp_df[cmp_df["metric_group"] == "transcript"].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["baseline_value", "data_type", "assembler_id", "test_id"]).reset_index(drop=True)
    sub["rank_idx"] = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, color in (("rf", "#1f77b4"), ("xgb", "#ff7f0e")):
        m = sub[sub["model"] == model]
        if m.empty:
            continue
        ax.scatter(m["rank_idx"], m["model_value"], s=14, color=color, label=f"{model.upper()} model", alpha=0.85)
    ax.scatter(sub["rank_idx"], sub["baseline_value"], s=10, color="#444444", label="Baseline", alpha=0.65)
    for _, r in sub.iterrows():
        ax.plot([r["rank_idx"], r["rank_idx"]], [r["baseline_value"], r["model_value"]], color="gray", alpha=0.15)
    ax.set_xlabel("Runs (sorted by baseline)")
    ax.set_ylabel("Transcript metric value")
    ax.set_title("Transcript model vs baseline across cross-annotation runs")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_tables(cmp_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(outdir / "cross_annotation_comparison_long.csv", index=False)
    agg = (
        cmp_df.groupby(["metric_group", "data_type", "assembler_id", "model"], dropna=False)
        .agg(
            n=("delta_vs_baseline", "size"),
            win_rate=("win_vs_baseline", "mean"),
            win_rate_pct=("win_vs_baseline", lambda x: float(np.mean(x) * 100.0)),
            delta_median=("delta_vs_baseline", "median"),
            delta_mean=("delta_vs_baseline", "mean"),
            rel_median=("relative_gain", "median"),
            rel_pct_median=("relative_gain_pct", "median"),
            rel_pct_mean=("relative_gain_pct", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(outdir / "cross_annotation_comparison_aggregates.csv", index=False)

    # Per-case percentage summary (train/test annotation case-level, already aggregator-friendly).
    by_case = (
        cmp_df.groupby(
            ["metric_group", "data_type", "assembler_id", "model", "train_annotation", "test_annotation"],
            dropna=False,
        )
        .agg(
            n=("delta_vs_baseline", "size"),
            win_rate_pct=("win_vs_baseline", lambda x: float(np.mean(x) * 100.0)),
            rel_pct_median=("relative_gain_pct", "median"),
            rel_pct_mean=("relative_gain_pct", "mean"),
            delta_median=("delta_vs_baseline", "median"),
        )
        .reset_index()
    )
    by_case.to_csv(outdir / "cross_annotation_percentage_by_case.csv", index=False)


def run_plotting(root: Path, outdir: Path) -> int:
    raw = load_all_summaries(root)
    cmp_df = to_comparison_table(raw)
    write_summary_tables(cmp_df, outdir)

    metrics = sorted(cmp_df["metric_group"].dropna().astype(str).unique())
    for metric in metrics:
        for model in ("rf", "xgb"):
            _heatmap_winrate(cmp_df, metric, model, outdir / f"{metric}_winrate_{model}.png")
            _pct_gain_heatmap(cmp_df, metric, model, outdir / f"{metric}_pct_gain_{model}.png")
        _delta_boxplot(cmp_df, metric, outdir / f"{metric}_delta_distribution.png")
    if "transcript" in metrics:
        _transcript_ranked_scatter(cmp_df, outdir / "transcript_ranked_run_scatter.png")
    print(f"[telos_v2] wrote plots and tables under: {outdir.resolve()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Plot cross-annotation benchmark summaries.")
    p.add_argument("--root", type=Path, default=ROOT, help="Root directory containing per-run folders.")
    p.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory for plots and summary tables.")
    args = p.parse_args()
    return run_plotting(args.root.resolve(), args.outdir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

