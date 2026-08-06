#!/usr/bin/env python3
"""Write Telos bundle_manifest.yaml (schema 1.0) for a finished sample work_dir."""

from __future__ import annotations

import argparse
from pathlib import Path


def _q(p: Path) -> str:
    s = str(p.resolve())
    if any(c in s for c in (":", "#", "'", '"')) or s.startswith(" "):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--ref-id", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--genome-fasta", type=Path, required=True)
    ap.add_argument("--ref-gtf", type=Path, required=True)
    ap.add_argument("--aligner", default="hisat2")
    args = ap.parse_args()

    wd = args.work_dir.resolve()
    bam = wd / "align" / "aln.sorted.bam"
    st = wd / "stringtie.gtf"
    asm, sec_gtf = (
        ("isoquant", wd / "isoquant.gtf")
        if (wd / "isoquant.gtf").is_file()
        else ("scallop2", wd / "scallop2.gtf")
    )
    tmap_st = wd / "stringtie.stringtie.gtf.tmap"
    tmap_sec = wd / f"{asm}.{asm}.gtf.tmap"

    lines = [
        "# Telos bundle_manifest (schema 1.0)",
        'schema_version: "1.0"',
        f"bundle_id: {args.ref_id}_{args.sample_id}_rnaseq",
        f"sample_id: {args.sample_id}",
        f"ref_id: {args.ref_id}",
        f"genome_fasta: {_q(args.genome_fasta)}",
        f"ref_annotation_gtf: {_q(args.ref_gtf)}",
        "",
        "alignments:",
        f"  sorted_bam: {_q(bam)}",
        '  bam_index: ""',
        f"  aligner: {args.aligner}",
        '  aligner_command: ""',
        "",
        "assemblies:",
        "  - assembler_id: stringtie",
        f"    gtf: {_q(st)}",
        "    gffcompare:",
        f"      tmap: {_q(tmap_st)}",
        '      refmap: ""',
        '      stats: ""',
        "      prefix: stringtie",
        f"  - assembler_id: {asm}",
        f"    gtf: {_q(sec_gtf)}",
        "    gffcompare:",
        f"      tmap: {_q(tmap_sec)}",
        '      refmap: ""',
        '      stats: ""',
        f"      prefix: {asm}",
        "",
        "provenance:",
        '  created_at: ""',
        "  fastq: []",
        "  tools: {}",
    ]
    out = wd / "bundle_manifest.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
