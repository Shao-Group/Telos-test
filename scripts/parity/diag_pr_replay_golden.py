#!/usr/bin/env python
"""Replay transcript PR on a reference ranked TSV with the current pipeline.

Paths come from env (or CLI). Example:

  export TELOS_BUNDLES_ROOT=$PWD/data/bundles
  export REF_RUNS_ROOT=/path/to/reference/runs
  export GFFCOMPARE=gffcompare
  python scripts/parity/diag_pr_replay_golden.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from telos_repro.evaluation.transcript_pr_pipeline import run_transcript_pr_benchmark


def main() -> None:
    bundles = Path(os.environ.get("TELOS_BUNDLES_ROOT", "data/bundles"))
    ref_runs = Path(os.environ.get("REF_RUNS_ROOT", ""))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gold-test-dir",
        type=Path,
        default=(
            ref_runs
            / "cross_annotation_repro/sr__train_gencode__test_gencode/tests/SRR307911__stringtie"
            if str(ref_runs)
            else None
        ),
    )
    ap.add_argument(
        "--assembly-gtf",
        type=Path,
        default=bundles / "GRCh38_gencode49/sr/SRR307911/stringtie.gtf",
    )
    ap.add_argument(
        "--ref-gtf",
        type=Path,
        default=Path("genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf"),
    )
    ap.add_argument(
        "--gffcompare-bin",
        default=os.environ.get("GFFCOMPARE") or None,
    )
    ap.add_argument("--out", type=Path, default=Path("runs/parity_diag/pr_replay_golden_ranked"))
    args = ap.parse_args()
    if args.gold_test_dir is None:
        raise SystemExit("Set --gold-test-dir or REF_RUNS_ROOT")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pr = run_transcript_pr_benchmark(
        assembly_gtf=args.assembly_gtf,
        ref_gtf=args.ref_gtf,
        ranked_tsv=args.gold_test_dir / "predictions/transcripts.ranked.rf.tsv",
        reports_pr_dir=out,
        work_rel="work_replay_rf",
        prefix="transcript_pr_rf",
        gffcompare_bin=args.gffcompare_bin,
        measure="cov",
        score_col="pred_prob",
        plot=False,
        filter_validation_chroms=True,
        autosome_train_range=(1, 10),
        save_pr_tables=True,
        ephemeral_workdir=False,
    )
    print("REPLAY model", pr["transcript_pr_auc_model"])
    print("REPLAY base ", pr["transcript_pr_auc_baseline"])
    print("meta", {k: pr[k] for k in pr if k.startswith("transcript_pr_") and "auc" not in k})


if __name__ == "__main__":
    main()
