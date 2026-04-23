"""
Glue between :mod:`telos_v2.benchmark.orchestrator` and :mod:`telos_v2.evaluation.transcript_pr_pipeline`.

Resolves default argument wiring (workdir naming, column names, ephemeral mode) so the orchestrator
stays linear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telos_v2.evaluation.transcript_pr_pipeline import run_transcript_pr_benchmark


def merge_transcript_pr_into_row(row: dict[str, Any], pr: dict[str, Any], backend_suffix: str) -> None:
    """
    Copy PR result dict keys into ``row`` with a ``_{backend_suffix}`` suffix.

    Transcript PR metrics are kept under explicit ``transcript_pr_*`` columns; Stage II AUPR columns
    are computed separately from ranked TSV × tmap labels (see orchestrator).
    """
    for k, v in pr.items():
        row[f"{k}_{backend_suffix}"] = v


def run_backend_transcript_pr(
    *,
    assembly_gtf: Path,
    ref_gtf: Path,
    ranked_tsv: Path,
    reports_pr_dir: Path,
    test_id: str,
    backend_suffix: str,
    gffcompare_bin: str | None,
    measure: str,
    plot: bool,
    save_pr_tables: bool,
    chromosomes_path: Path | None,
    filter_validation_chroms: bool,
    autosome_train_range: tuple[int, int] | None,
    ephemeral_workdir: bool,
) -> dict[str, Any]:
    """
    Run :func:`~telos_v2.evaluation.transcript_pr_pipeline.run_transcript_pr_benchmark` with benchmark defaults.

    Fixes ``score_col`` to ``pred_prob``, namespaces workdir/plot filenames by ``test_id`` and
    ``backend_suffix``, and passes chrom filtering / gffcompare / ephemeral flags through unchanged.
    """
    return run_transcript_pr_benchmark(
        assembly_gtf=assembly_gtf,
        ref_gtf=ref_gtf,
        ranked_tsv=ranked_tsv,
        reports_pr_dir=reports_pr_dir,
        work_rel=f"work_{test_id}_{backend_suffix}",
        prefix=f"transcript_pr_{backend_suffix}",
        gffcompare_bin=gffcompare_bin,
        measure=measure,
        score_col="pred_prob",
        plot=plot,
        plot_filename=f"transcript_pr_{backend_suffix}.png",
        chromosomes_file=chromosomes_path,
        filter_validation_chroms=filter_validation_chroms,
        autosome_train_range=autosome_train_range,
        save_pr_tables=save_pr_tables,
        ephemeral_workdir=ephemeral_workdir,
    )
