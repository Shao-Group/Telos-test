#!/usr/bin/env python3
"""Convert GTF exon records into BED12 (one row per transcript) for RSeQC infer_experiment.py."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def _parse_attrs(field9: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in field9.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part and not re.match(r'^\S+\s+"', part):
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
            continue
        m = re.match(r'(\S+)\s+"([^"]*)"', part)
        if m:
            out[m.group(1)] = m.group(2)
        else:
            m2 = re.match(r"(\S+)\s+(\S+)", part)
            if m2:
                out[m2.group(1)] = m2.group(2).strip('"')
    return out


def _collect_exons(path: str) -> Dict[str, Tuple[str, str, List[Tuple[int, int]]]]:
    """transcript_id -> (chrom, strand, sorted unique exons as (start,end) 1-based inclusive GTF)."""
    by_tid: Dict[str, Tuple[str, str, List[Tuple[int, int]]]] = {}
    exon_sets: Dict[str, set] = defaultdict(set)

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _source, feature, start_s, end_s, _score, strand, _frame, attrs = parts
            if feature != "exon":
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start >= end or strand not in "+-":
                continue
            ad = _parse_attrs(attrs)
            tid = ad.get("transcript_id") or ad.get("Parent")
            if not tid:
                continue
            key = (start, end)
            if key in exon_sets[tid]:
                continue
            exon_sets[tid].add(key)
            if tid not in by_tid:
                by_tid[tid] = (chrom, strand, [])
            else:
                c0, s0, _ = by_tid[tid]
                if c0 != chrom or s0 != strand:
                    sys.stderr.write(f"warn: transcript {tid} spans inconsistent chrom/strand; skipping extras\n")
                    continue
            by_tid[tid][2].append((start, end))

    for tid, (_c, _s, exons) in by_tid.items():
        exons.sort()
    return by_tid


def _to_bed12(tid: str, chrom: str, strand: str, exons: List[Tuple[int, int]]) -> str:
    if not exons:
        return ""
    tx_start = min(a for a, _b in exons)
    tx_end = max(b for _a, b in exons)
    bed_start = tx_start - 1
    bed_end = tx_end
    sizes = [str(b - a + 1) for a, b in exons]
    starts = [str((a - 1) - bed_start) for a, b in exons]
    block_count = len(exons)
    thick_start = bed_start
    thick_end = bed_end
    rgb = "0,0,0"
    return (
        f"{chrom}\t{bed_start}\t{bed_end}\t{tid}\t0\t{strand}\t{thick_start}\t{thick_end}\t{rgb}\t"
        f"{block_count}\t{','.join(sizes)}\t{','.join(starts)}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gtf", help="Input GTF (exon lines with transcript_id)")
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output BED12 path (default: stdout)",
    )
    args = ap.parse_args()

    by_tid = _collect_exons(args.gtf)
    rows = []
    for tid, (chrom, strand, exons) in by_tid.items():
        if not exons:
            continue
        line = _to_bed12(tid, chrom, strand, exons)
        if line:
            rows.append((chrom, int(line.split("\t", 3)[1]), line))

    rows.sort(key=lambda x: (x[0], x[1]))

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        for _c, _s, line in rows:
            out.write(line)
    finally:
        if args.output != "-":
            out.close()


if __name__ == "__main__":
    main()
