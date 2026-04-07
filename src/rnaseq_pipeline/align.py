"""FASTQ → coordinate-sorted, indexed BAM (minimap2 or HISAT2)."""

from __future__ import annotations

import logging
import os
import shlex
import tempfile
from pathlib import Path

from rnaseq_pipeline.config import LibraryPreset, RnaseqToolConfig, minimap2_ax_and_extra
from rnaseq_pipeline.exec_env import conda_run_cmd
from rnaseq_pipeline.fastq_input import FastqInput

LOG = logging.getLogger(__name__)


def _bash_pipe(conda_env: str, bash_script: str, *, cwd: Path) -> None:
    """
    Run a bash pipeline inside ``conda run``.

    Do not use ``bash -c '<script>'``: conda ``run`` rewrites inner commands into a temp
    shell script and can split on ``--`` (e.g. HISAT2 ``--dta``), yielding ``--: command
    not found`` on a stray line. A real script file avoids that.
    """
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="rnaseq_align_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(bash_script)
            if not bash_script.endswith("\n"):
                f.write("\n")
        os.chmod(path, 0o700)
        conda_run_cmd(conda_env, ["bash", path], cwd=cwd, check=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def align_to_sorted_bam(
    fq: FastqInput,
    ref_fasta: Path,
    out_bam: Path,
    preset: LibraryPreset,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> Path:
    """
    Produce coordinate-sorted BAM at ``out_bam`` and ``.bai`` beside it.

    - **NANOPORE / PACBIO**: minimap2 (see ``minimap2_ax_and_extra``) → samtools sort.
    - **SHORT_PAIRED**: HISAT2 + ``hisat2_extra_args`` (default ``--dta``) → samtools sort.
    """
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    ref_fasta = ref_fasta.resolve()
    if not ref_fasta.is_file():
        raise FileNotFoundError(ref_fasta)

    t = cfg.threads_align
    mm = shlex.quote(str(cfg.minimap2))
    st = shlex.quote(cfg.samtools)
    ref_q = shlex.quote(str(ref_fasta))
    out_q = shlex.quote(str(out_bam))

    if preset in (LibraryPreset.NANOPORE, LibraryPreset.PACBIO):
        ax, extra = minimap2_ax_and_extra(cfg, preset)
        ax_q = shlex.quote(ax)
        extra_q = " ".join(shlex.quote(x) for x in extra)
        fqs = [shlex.quote(str(p)) for p in fq.r1]
        if fq.is_paired:
            raise ValueError(
                "Paired FASTQ with nanopore/pacbio preset is unusual; use preset=short_paired "
                "or provide single-end long-read FASTQ."
            )
        query = " ".join(fqs)
        mid = f"{extra_q} " if extra_q else ""
        script = (
            f"set -euo pipefail; {mm} -ax {ax_q} {mid}-t {t} {ref_q} {query} "
            f"| {st} sort --threads {t} -o {out_q} -"
        )
        LOG.info("minimap2 -ax %s %s| samtools sort -> %s", ax, extra, out_bam)
        _bash_pipe(cfg.conda_env, script, cwd=work_dir)
    elif preset == LibraryPreset.SHORT_PAIRED:
        if cfg.hisat2_index is None:
            raise ValueError("hisat2_index must be set on RnaseqToolConfig for SHORT_PAIRED")
        hx = shlex.quote(str(cfg.hisat2))
        idx = shlex.quote(str(cfg.hisat2_index))
        hs_parts: list[str] = []
        if cfg.hisat2_rna_strandness:
            hs_parts.extend(["--rna-strandness", cfg.hisat2_rna_strandness])
        hs_parts.extend(cfg.hisat2_extra_args)
        tail = " ".join(shlex.quote(x) for x in hs_parts)
        if fq.is_paired and fq.r2:
            r1 = [shlex.quote(str(p)) for p in fq.r1]
            r2 = [shlex.quote(str(p)) for p in fq.r2]
            u1 = ",".join(r1)
            u2 = ",".join(r2)
            script = (
                f"set -euo pipefail; {hx} -x {idx} -p {t} -1 {u1} -2 {u2} {tail} "
                f"| {st} sort --threads {t} -o {out_q} -"
            )
        else:
            u = ",".join(shlex.quote(str(p)) for p in fq.r1)
            script = (
                f"set -euo pipefail; {hx} -x {idx} -p {t} -U {u} {tail} "
                f"| {st} sort --threads {t} -o {out_q} -"
            )
        LOG.info("hisat2 strand=%s extras=%s | samtools sort -> %s", cfg.hisat2_rna_strandness, cfg.hisat2_extra_args, out_bam)
        _bash_pipe(cfg.conda_env, script, cwd=work_dir)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    index_bam(out_bam, cfg)
    return out_bam


def index_bam(bam: Path, cfg: RnaseqToolConfig) -> None:
    conda_run_cmd(
        cfg.conda_env,
        [cfg.samtools, "index", str(bam)],
        cwd=bam.parent,
        check=True,
    )
