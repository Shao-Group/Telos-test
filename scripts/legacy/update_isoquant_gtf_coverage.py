#!/usr/bin/env python3
"""Bulk-update IsoQuant GTF transcript coverage fields from TPM tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from rnaseq_pipeline.isoquant_tpm_update import (
    discover_isoquant_pairs,
    update_isoquant_gtf_coverage_from_tpm,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Find isoquant.gtf + isoquant_transcript_model_tpm.tsv pairs recursively and "
            "rewrite transcript coverage attributes from TPM values."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/bundles"),
        help="Root directory to scan for bundle/sample outputs.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite isoquant.gtf directly. By default writes a sidecar file.",
    )
    p.add_argument(
        "--output-name",
        default="isoquant.with_tpm_coverage.gtf",
        help="Output filename used when not running with --in-place.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of discovered pairs to process (0 = no limit).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    pairs = discover_isoquant_pairs(root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"[isoquant-tpm-update] no pairs found under {root}")
        return 1

    print(f"[isoquant-tpm-update] discovered pairs: {len(pairs)}")
    total_tx = 0
    total_updated = 0
    total_missing = 0

    for i, (gtf, tpm) in enumerate(pairs, start=1):
        out_gtf = gtf if args.in_place else gtf.with_name(args.output_name)
        result = update_isoquant_gtf_coverage_from_tpm(gtf, tpm, out_gtf)
        total_tx += result.transcript_lines
        total_updated += result.updated_from_tpm
        total_missing += result.missing_in_tpm
        print(
            f"[{i}/{len(pairs)}] {gtf.parent} -> {out_gtf.name} | "
            f"tx={result.transcript_lines} updated={result.updated_from_tpm} "
            f"missing_tpm={result.missing_in_tpm}"
        )

    print("[isoquant-tpm-update] done")
    print(
        f"[isoquant-tpm-update] totals: tx={total_tx} updated={total_updated} "
        f"missing_tpm={total_missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
