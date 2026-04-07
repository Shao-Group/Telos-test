"""StringTie3, IsoQuant, Scallop2, optional gtfformat — mirrors generate-*.sh."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from rnaseq_pipeline.config import LibraryPreset, RnaseqToolConfig
from rnaseq_pipeline.exec_env import conda_run_cmd

LOG = logging.getLogger(__name__)


def run_stringtie(
    bam: Path,
    out_gtf: Path,
    preset: LibraryPreset,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> Path:
    """StringTie3: ``-L`` for long-read presets only."""
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        str(cfg.stringtie),
        "-p",
        str(cfg.threads_assembly),
    ]
    if preset in (LibraryPreset.NANOPORE, LibraryPreset.PACBIO):
        cmd.append("-L")
    cmd.extend(["-o", str(out_gtf), str(bam)])
    conda_run_cmd(cfg.conda_env, cmd, cwd=work_dir, check=True)
    return out_gtf


def _run_isoquant_once(
    bam: Path,
    ref_fasta: Path,
    out_dir: Path,
    data_type: str,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
    extra_args: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        cfg.isoquant_script,
        "--threads",
        str(cfg.threads_isoquant),
        "--reference",
        str(ref_fasta),
        "--bam",
        str(bam),
        "--data_type",
        data_type,
        "-o",
        str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    conda_run_cmd(cfg.conda_env, cmd, cwd=work_dir, check=True)


def _copy_isoquant_gtf_tpm(work_isoquant: Path, dest_gtf: Path, dest_tpm: Path) -> None:
    src_gtf = work_isoquant / "OUT" / "OUT.transcript_models.gtf"
    src_tpm = work_isoquant / "OUT" / "OUT.transcript_model_tpm.tsv"
    if not src_gtf.is_file():
        raise FileNotFoundError(f"IsoQuant GTF not found: {src_gtf}")
    shutil.copy2(src_gtf, dest_gtf)
    if src_tpm.is_file():
        shutil.copy2(src_tpm, dest_tpm)


def run_isoquant(
    bam: Path,
    ref_fasta: Path,
    out_gtf: Path,
    out_tpm: Path,
    preset: LibraryPreset,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> Path:
    """Single IsoQuant run (default model); writes ``out_gtf`` and TPM table."""
    if preset == LibraryPreset.SHORT_PAIRED:
        raise ValueError("IsoQuant not used for SHORT_PAIRED preset")

    data_type = "nanopore" if preset == LibraryPreset.NANOPORE else "pacbio"
    w = work_dir / "isoquant_out"
    if w.exists():
        shutil.rmtree(w)
    _run_isoquant_once(bam, ref_fasta, w, data_type, cfg, work_dir=work_dir, extra_args=None)
    _copy_isoquant_gtf_tpm(w, out_gtf, out_tpm)
    shutil.rmtree(w)
    return out_gtf


def run_gtfformat_update_tpm(
    gtf: Path,
    tpm: Path,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> None:
    if cfg.gtfformat is None:
        LOG.info("gtfformat not configured; skipping TPM merge")
        return
    if not tpm.is_file():
        raise FileNotFoundError(tpm)
    cmd = [str(cfg.gtfformat), "update-tpm", str(gtf), str(tpm), str(gtf)]
    conda_run_cmd(cfg.conda_env, cmd, cwd=work_dir, check=True)


def run_scallop2(
    bam: Path,
    out_gtf: Path,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> Path:
    cmd = [cfg.scallop2_cmd, "-i", str(bam), "-o", str(out_gtf)]
    conda_run_cmd(cfg.conda_env, cmd, cwd=work_dir, check=True)
    return out_gtf
