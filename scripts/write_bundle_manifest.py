#!/usr/bin/env python3
"""Emit a Telos bundle_manifest YAML for one rnaseq_pipeline work_dir (schema 1.0)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _quote(p: Path) -> str:
    s = str(p.resolve())
    if any(c in s for c in (":", "#", "'", '"')) or s.startswith(" "):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


# YAML empty string (f-string expressions cannot contain backslashes)
_YAML_EMPTY = '""'


def _quote_optional(p: Path | None) -> str:
    return _quote(p) if p is not None else _YAML_EMPTY


def _find_tmap(gffcmp: Path, query_gtf: Path) -> Path | None:
    """
    Resolve GFFCompare .tmap for a query GTF.

    gffcompare writes ``<outprefix>.<query_basename>.tmap`` in the **directory of the
    query file**, not under ``gffcmp/``. Search there first, then ``gffcmp/`` for older layouts.
    """
    base = query_gtf.name
    for d in (query_gtf.parent, gffcmp):
        hits = sorted(d.glob(f"*.{base}.tmap"))
        if hits:
            return hits[0]
        hits = sorted(d.glob(f"*{base}.tmap"))
        if hits:
            return hits[0]
    return None


def _find_gffcmp_sidecar(gffcmp: Path, prefix: str, suffix: str) -> Path | None:
    """
    Resolve sidecar files for one gffcompare prefix.

    - ``.stats`` is typically under ``gffcmp/<prefix>.stats``.
    - ``.refmap`` is often written beside query GTFs as ``<prefix>.*.refmap``.
    """
    p = gffcmp / f"{prefix}.{suffix}"
    if p.is_file():
        return p
    if suffix == "refmap":
        for d in (gffcmp.parent, gffcmp):
            hits = sorted(d.glob(f"{prefix}.*.refmap"))
            if hits:
                return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--ref-id", type=str, required=True, help="e.g. GRCh38_gencode49")
    ap.add_argument("--sample-id", type=str, required=True)
    ap.add_argument("--genome-fasta", type=Path, required=True)
    ap.add_argument("--ref-gtf", type=Path, required=True)
    ap.add_argument("--aligner", type=str, default="hisat2", help="hisat2 or minimap2")
    args = ap.parse_args()

    wd: Path = args.work_dir.resolve()
    bam = wd / "align" / "aln.sorted.bam"
    if not bam.is_file():
        raise SystemExit(f"Missing sorted BAM: {bam}")

    st = wd / "stringtie.gtf"
    if not st.is_file():
        raise SystemExit(f"Missing {st}")

    iso = wd / "isoquant.gtf"
    sc2 = wd / "scallop2.gtf"
    if iso.is_file():
        secondary = ("isoquant", iso)
    elif sc2.is_file():
        secondary = ("scallop2", sc2)
    else:
        raise SystemExit(f"Missing isoquant.gtf or scallop2.gtf under {wd}")

    gffcmp = wd / "gffcmp"
    if not gffcmp.is_dir():
        raise SystemExit(f"Missing gffcmp/ under {wd}")

    tmap_st = _find_tmap(gffcmp, st)
    tmap_sec = _find_tmap(gffcmp, secondary[1])
    if not tmap_st or not tmap_sec:
        raise SystemExit(
            f"Could not resolve .tmap for stringtie / {secondary[0]} "
            f"(looked in {wd} and {gffcmp}; expected e.g. *.{st.name}.tmap beside the query GTF)"
        )

    stats_st = _find_gffcmp_sidecar(gffcmp, "stringtie", "stats")
    stats_sec = _find_gffcmp_sidecar(gffcmp, secondary[0], "stats")
    refmap_st = _find_gffcmp_sidecar(gffcmp, "stringtie", "refmap")
    refmap_sec = _find_gffcmp_sidecar(gffcmp, secondary[0], "refmap")

    bundle_id = f"{args.ref_id}_{args.sample_id}_rnaseq"

    lines: list[str] = [
        "# Telos bundle_manifest (schema 1.0) — paths absolute",
        'schema_version: "1.0"',
        f"bundle_id: {bundle_id}",
        f"sample_id: {args.sample_id}",
        f"ref_id: {args.ref_id}",
        f"genome_fasta: {_quote(args.genome_fasta)}",
        f"ref_annotation_gtf: {_quote(args.ref_gtf)}",
        "",
        "alignments:",
        f"  sorted_bam: {_quote(bam)}",
        '  bam_index: ""',
        f"  aligner: {args.aligner}",
        '  aligner_command: ""',
        "",
        "assemblies:",
        "  - assembler_id: stringtie",
        f"    gtf: {_quote(st)}",
        "    gffcompare:",
        f"      tmap: {_quote(tmap_st)}",
        f"      refmap: {_quote_optional(refmap_st)}",
        f"      stats: {_quote_optional(stats_st)}",
        "      prefix: stringtie",
        f"  - assembler_id: {secondary[0]}",
        f"    gtf: {_quote(secondary[1])}",
        "    gffcompare:",
        f"      tmap: {_quote(tmap_sec)}",
        f"      refmap: {_quote_optional(refmap_sec)}",
        f"      stats: {_quote_optional(stats_sec)}",
        f"      prefix: {secondary[0]}",
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
