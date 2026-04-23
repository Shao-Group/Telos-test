"""
Evaluation helpers: sklearn PR diagnostics, abundance-baseline PR math, and gffcompare transcript PR.
"""

from telos_v2.evaluation.pr_ranking import (
    PRCurve,
    PRComparisonResult,
    compute_pr_curves_vs_baseline,
    plot_pr_comparison,
    run_transcript_pr_benchmark_artifacts,
    site_pr_vs_baseline,
    transcript_pr_vs_baseline_from_files,
    write_pr_comparison_outputs,
)

__all__ = [
    "PRCurve",
    "PRComparisonResult",
    "compute_pr_curves_vs_baseline",
    "plot_pr_comparison",
    "run_transcript_pr_benchmark_artifacts",
    "site_pr_vs_baseline",
    "transcript_pr_vs_baseline_from_files",
    "write_pr_comparison_outputs",
]
