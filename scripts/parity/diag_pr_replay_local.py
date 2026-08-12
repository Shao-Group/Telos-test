#!/usr/bin/env python
"""Replay transcript PR on a local ranked TSV from a prior parity/bench run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from telos_repro.evaluation.transcript_pr_pipeline import run_transcript_pr_benchmark


def main() -> None:
    bundles = Path(os.environ.get("TELOS_BUNDLES_ROOT", "data/bundles"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--local-test-dir",
        type=Path,
        default=Path(
            "runs/parity_tier1_golden/telos/tests/SRR307911__stringtie"
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
    ap.add_argument("--out", type=Path, default=Path("runs/parity_diag/pr_replay_local_ranked"))
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pr = run_transcript_pr_benchmark(
        assembly_gtf=args.assembly_gtf,
        ref_gtf=args.ref_gtf,
        ranked_tsv=args.local_test_dir / "predictions/transcripts.ranked.rf.tsv",
        reports_pr_dir=out,
        work_rel="work_replay_rf",
        prefix="transcript_pr_rf",
        gffcompare_bin=args.gffcompare_bin,
        measure="cov",
        plot=False,
        filter_validation_chroms=True,
        autosome_train_range=(1, 10),
        save_pr_tables=True,
        ephemeral_workdir=True,
    )
    print("LOCAL replay RF", pr["transcript_pr_auc_model"])


if __name__ == "__main__":
    main()
