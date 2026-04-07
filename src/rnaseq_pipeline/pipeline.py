"""End-to-end FASTQ → BAM → assemblies → GFFCompare."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from rnaseq_pipeline.align import align_to_sorted_bam
from rnaseq_pipeline.assemble import (
    run_gtfformat_update_tpm,
    run_isoquant,
    run_scallop2,
    run_stringtie,
)
from rnaseq_pipeline.config import LibraryPreset, RnaseqToolConfig
from rnaseq_pipeline.fastq_input import discover_fastq
from rnaseq_pipeline.gffcompare_run import run_gffcompare_batch

LOG = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    work_dir: Path
    bam: Path
    stringtie_gtf: Path
    """IsoQuant or Scallop2 GTF (depends on preset)."""
    assembler_secondary_gtf: Optional[Path]
    isoquant_tpm: Optional[Path]
    gffcmp_dir: Path
    gffcompare_pairs: List[Tuple[Path, str]] = field(default_factory=list)


class RnaseqAssemblyPipeline:
    """
    Orchestrates alignment, StringTie, second assembler (IsoQuant or Scallop2),
    optional gtfformat, and GFFCompare — aligned with ``generate-*assemblies.sh``.
    """

    def __init__(self, cfg: Optional[RnaseqToolConfig] = None) -> None:
        self.cfg = cfg or RnaseqToolConfig()

    def run(
        self,
        *,
        fastq_path: Path,
        work_dir: Path,
        ref_fasta: Path,
        ref_gtf: Path,
        preset: LibraryPreset,
        skip_align: bool = False,
        existing_bam: Optional[Path] = None,
        bam_name: str = "aln.sorted.bam",
    ) -> PipelineResult:
        """
        :param fastq_path: File or directory (see ``discover_fastq``).
        :param work_dir: All outputs written here (created if missing).
        :param skip_align: If True, use ``existing_bam`` (must be sorted + indexed).
        :param existing_bam: Required when ``skip_align`` is True.
        """
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        align_dir = work_dir / "align"
        align_dir.mkdir(parents=True, exist_ok=True)
        bam_path = align_dir / bam_name

        if skip_align:
            if existing_bam is None:
                raise ValueError("existing_bam is required when skip_align=True")
            existing_bam = existing_bam.resolve()
            if not existing_bam.is_file():
                raise FileNotFoundError(existing_bam)
            bam_path = existing_bam
            bai = bam_path.parent / (bam_path.name + ".bai")
            if not bai.is_file():
                LOG.warning("BAM index not found beside %s; indexing now.", bam_path)
                from rnaseq_pipeline.align import index_bam

                index_bam(bam_path, self.cfg)
        else:
            fq = discover_fastq(fastq_path)
            align_to_sorted_bam(
                fq,
                ref_fasta,
                bam_path,
                preset,
                self.cfg,
                work_dir=work_dir,
            )

        st_gtf = work_dir / "stringtie.gtf"
        run_stringtie(bam_path, st_gtf, preset, self.cfg, work_dir=work_dir)

        sec_gtf: Optional[Path] = None
        tpm: Optional[Path] = None
        gff_pairs: List[Tuple[Path, str]] = [(st_gtf, "stringtie")]

        if preset in (LibraryPreset.NANOPORE, LibraryPreset.PACBIO):
            sec_gtf = work_dir / "isoquant.gtf"
            tpm = work_dir / "isoquant_transcript_model_tpm.tsv"
            run_isoquant(
                bam_path,
                ref_fasta,
                sec_gtf,
                tpm,
                preset,
                self.cfg,
                work_dir=work_dir,
            )
            run_gtfformat_update_tpm(sec_gtf, tpm, self.cfg, work_dir=work_dir)
            gff_pairs.append((sec_gtf, "isoquant"))
        else:
            sec_gtf = work_dir / "scallop2.gtf"
            run_scallop2(bam_path, sec_gtf, self.cfg, work_dir=work_dir)
            gff_pairs.append((sec_gtf, "scallop2"))

        gff_dir = run_gffcompare_batch(gff_pairs, ref_gtf, self.cfg, work_dir=work_dir)

        return PipelineResult(
            work_dir=work_dir,
            bam=bam_path,
            stringtie_gtf=st_gtf,
            assembler_secondary_gtf=sec_gtf,
            isoquant_tpm=tpm,
            gffcmp_dir=gff_dir,
            gffcompare_pairs=gff_pairs,
        )
