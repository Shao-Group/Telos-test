"""Evaluation helpers: PR curves vs abundance baselines (gtfcuff-style ranking)."""

from telos_v2.evaluation.pr_ranking import (
    PRCurve,
    PRComparisonResult,
    compute_pr_curves_vs_baseline,
    plot_pr_comparison,
    run_transcript_pr_benchmark_artifacts,
    site_pr_vs_baseline,
    transcript_pr_vs_baseline_from_files,
    write_pr_comparison_csv,
    write_pr_comparison_outputs,
    write_pr_comparison_tsv,
)

__all__ = [
    "PRCurve",
    "PRComparisonResult",
    "compute_pr_curves_vs_baseline",
    "plot_pr_comparison",
    "run_transcript_pr_benchmark_artifacts",
    "site_pr_vs_baseline",
    "transcript_pr_vs_baseline_from_files",
    "write_pr_comparison_csv",
    "write_pr_comparison_outputs",
    "write_pr_comparison_tsv",
]
