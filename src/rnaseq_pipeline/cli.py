"""CLI: ``python -m rnaseq_pipeline`` with PYTHONPATH=src."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rnaseq_pipeline.config import LibraryPreset, NanoporeLibraryKind, RnaseqToolConfig
from rnaseq_pipeline.hisat2_index import build_hisat2_index
from rnaseq_pipeline.pipeline import RnaseqAssemblyPipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FASTQ → sorted BAM → StringTie/IsoQuant|Scallop2 → GFFCompare (Telos prep).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    idx = sub.add_parser(
        "build-hisat2-index",
        help="Run hisat2-build (optional GTF splice sites); same prefix as --hisat2-index for run.",
    )
    idx.add_argument(
        "--ref-fasta",
        type=Path,
        required=True,
        help="Reference genome FASTA (same assembly as Telos / GFFCompare).",
    )
    idx.add_argument(
        "--index-prefix",
        type=Path,
        required=True,
        help="Output index basename (no .ht2); pass this path to run --hisat2-index.",
    )
    idx.add_argument(
        "--ref-gtf",
        type=Path,
        default=None,
        help="Optional: reference GTF → extract splice sites + exons for better RNA-seq index.",
    )
    idx.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Where to write splice/exon txt files (default: parent of --index-prefix).",
    )
    idx.add_argument("--threads", type=int, default=8, help="hisat2-build -p threads.")
    idx.add_argument(
        "--conda-env",
        type=str,
        default="irtesam-berth",
        help="Conda env (default: irtesam-berth).",
    )
    idx.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")

    run = sub.add_parser("run", help="Run full pipeline for one sample")
    run.add_argument(
        "--preset",
        type=str,
        choices=[e.value for e in LibraryPreset],
        required=True,
        help="nanopore / pacbio / short_paired",
    )
    run.add_argument(
        "--fastq",
        type=Path,
        required=True,
        help="FASTQ file or directory of FASTQ (paired: *_1/*_2 or R1/R2).",
    )
    run.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Output directory for BAM, GTFs, gffcmp/.",
    )
    run.add_argument("--ref-fasta", type=Path, required=True, help="Reference genome FASTA.")
    run.add_argument(
        "--ref-gtf",
        type=Path,
        required=True,
        help="Reference annotation GTF for gffcompare -r (match genome build).",
    )
    run.add_argument(
        "--skip-align",
        action="store_true",
        help="Skip alignment; requires --bam (sorted BAM, will index if .bai missing).",
    )
    run.add_argument(
        "--bam",
        type=Path,
        default=None,
        help="Sorted BAM when using --skip-align.",
    )
    run.add_argument(
        "--conda-env",
        type=str,
        default="irtesam-berth",
        help="Conda environment for all tools (default: irtesam-berth).",
    )
    run.add_argument(
        "--isoquant-conda-env",
        type=str,
        default=None,
        help="Conda env for IsoQuant only (nanopore/pacbio). "
        "Default: $ISOQUANT_CONDA_ENV, else same as --conda-env. "
        "Use when main env has broken sqlite3 (ImportError sqlite3_deserialize).",
    )
    run.add_argument(
        "--isoquant-script",
        type=str,
        default=None,
        help="IsoQuant executable name in that env (e.g. isoquant or isoquant.py). "
        "Default: $ISOQUANT_SCRIPT, else 'isoquant'.",
    )
    run.add_argument(
        "--samtools",
        type=Path,
        default=None,
        help="samtools binary. Default: $SAMTOOLS, else $CONDA_PREFIX/bin/samtools if present, "
        "else lookup on PATH.",
    )
    run.add_argument("--threads-align", type=int, default=8)
    run.add_argument("--threads-assembly", type=int, default=8)
    run.add_argument("--threads-isoquant", type=int, default=32)
    run.add_argument(
        "--nanopore-library",
        type=str,
        choices=[e.value for e in NanoporeLibraryKind],
        default=NanoporeLibraryKind.CDNA.value,
        help="ONT only: cdna (default -ax splice) vs drna (splice -uf -k14). Ignored for other presets.",
    )
    run.add_argument(
        "--minimap2-ax",
        type=str,
        default=None,
        help="Override minimap2 -ax preset (e.g. splice, splice:hq). Default: preset-based.",
    )
    run.add_argument(
        "--minimap2-extra",
        type=str,
        action="append",
        default=None,
        help="Extra minimap2 args after -ax (repeatable), e.g. --minimap2-extra --secondary=no",
    )
    run.add_argument(
        "--hisat2-index",
        type=Path,
        default=None,
        help="HISAT2 index *prefix* for -x (same as hisat2-build output basename, no .ht2).",
    )
    run.add_argument(
        "--hisat2-strandness",
        type=str,
        choices=["RF", "FR", "F", "R"],
        default=None,
        help="HISAT2 --rna-strandness: paired-end RF/FR, single-end F/R. "
        "ENCODE long RNA (GSE30567) is directional; confirm with GEO or RSeQC infer_experiment.",
    )
    run.add_argument(
        "--hisat2-extra",
        type=str,
        action="append",
        default=None,
        help="Extra HISAT2 args before pipe (repeatable). Default includes --dta for StringTie.",
    )
    run.add_argument(
        "--gtfformat",
        type=Path,
        default=None,
        help="Path to gtfformat binary for IsoQuant TPM merge (optional).",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "build-hisat2-index":
        cfg = RnaseqToolConfig(conda_env=args.conda_env)
        prefix = build_hisat2_index(
            cfg,
            ref_fasta=args.ref_fasta,
            index_prefix=args.index_prefix,
            threads=args.threads,
            ref_gtf=args.ref_gtf,
            work_dir=args.staging_dir,
        )
        print("HISAT2 index ready. Use with:")
        print(f"  --hisat2-index {prefix}")
        return 0

    if args.command != "run":
        return 1

    if args.skip_align and not args.bam:
        print("--bam is required with --skip-align", file=sys.stderr)
        return 2

    mm_extra = list(args.minimap2_extra) if args.minimap2_extra else []
    hs_extra = list(args.hisat2_extra) if args.hisat2_extra is not None else None

    isoq_env = args.isoquant_conda_env or os.environ.get("ISOQUANT_CONDA_ENV")
    isoq_script = args.isoquant_script or os.environ.get("ISOQUANT_SCRIPT")
    cfg_kw: dict = dict(
        conda_env=args.conda_env,
        threads_align=args.threads_align,
        threads_assembly=args.threads_assembly,
        threads_isoquant=args.threads_isoquant,
        nanopore_library=NanoporeLibraryKind(args.nanopore_library),
        minimap2_ax_preset=args.minimap2_ax,
        minimap2_extra_args=mm_extra,
        hisat2_index=args.hisat2_index,
        hisat2_rna_strandness=args.hisat2_strandness,
        hisat2_extra_args=hs_extra if hs_extra is not None else RnaseqToolConfig().hisat2_extra_args,
        gtfformat=args.gtfformat,
    )
    if isoq_env:
        cfg_kw["isoquant_conda_env"] = isoq_env
    if isoq_script:
        cfg_kw["isoquant_script"] = isoq_script
    if args.samtools is not None:
        cfg_kw["samtools"] = str(args.samtools.resolve())
    cfg = RnaseqToolConfig(**cfg_kw)

    pipeline = RnaseqAssemblyPipeline(cfg)
    result = pipeline.run(
        fastq_path=args.fastq,
        work_dir=args.work_dir,
        ref_fasta=args.ref_fasta,
        ref_gtf=args.ref_gtf,
        preset=LibraryPreset(args.preset),
        skip_align=args.skip_align,
        existing_bam=args.bam,
    )

    print("Done.")
    print(f"  BAM:           {result.bam}")
    print(f"  StringTie:     {result.stringtie_gtf}")
    print(f"  Secondary:     {result.assembler_secondary_gtf}")
    print(f"  GFFCompare:    {result.gffcmp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
