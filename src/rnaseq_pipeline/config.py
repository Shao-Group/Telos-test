"""
Central configuration for tools, conda environment, and library-type presets.

Edit defaults here (or pass ``RnaseqToolConfig`` from code) so paths stay in one place.

**Aligner defaults (literature / tool docs)**

- **ONT cDNA (nanopore + cdna):** minimap2 ``-ax splice`` — general spliced alignment for
  Nanopore transcript reads (see minimap2 ``-x splice`` / RNA docs).
- **ONT direct RNA (nanopore + drna):** minimap2 ``-ax splice`` with ``-uf -k14`` — common
  recommendation for noisy dRNA (e.g. community notes and `minimap2` RNA examples).
- **PacBio (Iso-Seq / Kinnex-style RNA):** minimap2 ``-ax splice:hq`` — PacBio spliced
  long-read RNA preset in minimap2.
- **Short reads + StringTie:** HISAT2 with ``--dta`` so alignments include splice info
  StringTie expects (`HISAT2 --dta`, StringTie / Nature Protocols-style pipelines).

Override any of this via ``minimap2_ax_preset``, ``minimap2_extra_args``, or
``hisat2_extra_args`` when your library or protocol differs.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


def _default_conda_bin(exe_name: str) -> Path:
    """
    Prefer executable from active conda env (conda run sets CONDA_PREFIX).
    Fall back to bare name so PATH lookup still works.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / exe_name
        if candidate.is_file():
            return candidate
    return Path(exe_name)


def _default_samtools() -> str:
    """
    Prefer explicit SAMTOOLS, then samtools next to the active conda env (conda run sets
    CONDA_PREFIX), then PATH lookup. Bare ``samtools`` breaks inner ``bash -c`` pipes when
    PATH does not include the env bin (exit 127).
    """
    env = os.environ.get("SAMTOOLS")
    if env:
        return env
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / "samtools"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("samtools")
    return found if found else "samtools"


class LibraryPreset(str, Enum):
    """Which assembly workflow to run (mirrors the three shell scripts)."""

    NANOPORE = "nanopore"
    """StringTie ``-L`` + IsoQuant ``--data_type nanopore`` + optional gtfformat."""

    PACBIO = "pacbio"
    """StringTie ``-L`` + IsoQuant ``--data_type pacbio`` + optional gtfformat."""

    SHORT_PAIRED = "short_paired"
    """StringTie (no ``-L``) + Scallop2; no IsoQuant (``generate_sr_assemblies.sh``)."""


class NanoporeLibraryKind(str, Enum):
    """ONT chemistry for minimap2 defaults (only used with ``NANOPORE`` preset)."""

    CDNA = "cdna"
    """PCR/cDNA Nanopore RNA — default ``-ax splice``."""

    DRNA = "drna"
    """Direct RNA — default ``-ax splice`` plus ``-uf -k14`` (noisy reads)."""


@dataclass
class RnaseqToolConfig:
    """Executable paths and runtime defaults."""

    conda_env: str = "irtesam-berth"

    stringtie: Path = field(
        default_factory=lambda: Path(
            "/datadisk1/ixk5174/tools/stringtie-3.0.3.Linux_x86_64/stringtie"
        )
    )
    gffcompare: Path = field(
        default_factory=lambda: Path(
            "/datadisk1/ixk5174/tools/gffcompare-0.12.10.Linux_x86_64/gffcompare"
        )
    )

    minimap2: Path = field(
        default_factory=lambda: Path("/datadisk1/shared/tools/minimap2/minimap2")
    )
    hisat2: Path = field(
        default_factory=lambda: _default_conda_bin("hisat2")
    )
    hisat2_build: Path = field(
        default_factory=lambda: _default_conda_bin("hisat2-build")
    )
    hisat2_inspect: Path = field(
        default_factory=lambda: _default_conda_bin("hisat2-inspect")
    )
    samtools: str = field(default_factory=_default_samtools)

    isoquant_script: str = "isoquant.py"
    scallop2_cmd: str = "scallop2"

    gtfformat: Optional[Path] = None

    threads_align: int = 8
    threads_assembly: int = 8
    threads_isoquant: int = 32

    # Nanopore-only: which ONT library type (drives default minimap2 -ax / extra args)
    nanopore_library: NanoporeLibraryKind = NanoporeLibraryKind.CDNA

    # If set, overrides automatic -ax from nanopore_library / pacbio
    minimap2_ax_preset: Optional[str] = None
    """e.g. ``splice``, ``splice:hq`` — when None, chosen from preset + nanopore_library."""

    # Appended after -ax (e.g. ONT dRNA: -uf -k14); merged with preset defaults unless overridden
    minimap2_extra_args: List[str] = field(default_factory=list)

    # Prefix passed to ``hisat2 -x`` (same basename you used with ``hisat2-build``, no extension).
    # Example: ``hisat2-build ... -p 4 genome.fa grch38/grch38`` → use Path("grch38/grch38").
    hisat2_index: Optional[Path] = None

    # Paired RNA-seq only: ``RF``, ``FR``, or None (treat as unstranded for strandness).
    # ENCODE long RNA CSHL (e.g. GSE30567) describes directional libraries; confirm with GEO/SRA or RSeQC.
    hisat2_rna_strandness: Optional[str] = None

    # Default --dta for StringTie-compatible spliced alignments (Nature Protocols-style)
    hisat2_extra_args: List[str] = field(default_factory=lambda: ["--dta"])


def minimap2_ax_and_extra(cfg: "RnaseqToolConfig", preset: LibraryPreset) -> Tuple[str, List[str]]:
    """
    Return ``(-ax NAME, extra_args)`` for minimap2.

    PacBio: ``splice:hq``.
    Nanopore cDNA: ``splice``; Nanopore dRNA: ``splice`` + ``-uf -k14``; PacBio: ``splice:hq``.
    User overrides: ``minimap2_ax_preset`` and/or ``minimap2_extra_args``.
    """
    if cfg.minimap2_ax_preset is not None:
        ax = cfg.minimap2_ax_preset
        extra = list(cfg.minimap2_extra_args)
        return ax, extra

    if preset == LibraryPreset.PACBIO:
        return "splice:hq", list(cfg.minimap2_extra_args)

    if preset == LibraryPreset.NANOPORE:
        if cfg.nanopore_library == NanoporeLibraryKind.DRNA:
            return "splice", ["-uf", "-k14", *cfg.minimap2_extra_args]
        return "splice", list(cfg.minimap2_extra_args)

    raise ValueError("minimap2 defaults only apply to NANOPORE or PACBIO presets")
