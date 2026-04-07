"""Build HISAT2 genome index (plain or splice-aware from GTF)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rnaseq_pipeline.config import RnaseqToolConfig
from rnaseq_pipeline.exec_env import conda_run_cmd

LOG = logging.getLogger(__name__)


def _hisat2_dir(cfg: RnaseqToolConfig) -> Path:
    return cfg.hisat2.parent


def build_hisat2_index(
    cfg: RnaseqToolConfig,
    *,
    ref_fasta: Path,
    index_prefix: Path,
    threads: int,
    ref_gtf: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> Path:
    """
    Run ``hisat2-build`` into ``index_prefix`` (no ``.ht2`` suffix — same value as ``-x`` for alignment).

    If ``ref_gtf`` is set, runs ``extract_splice_sites.py`` and ``extract_exons.py`` from the
    same HISAT2 distribution and passes ``--ss`` / ``--exon`` to ``hisat2-build`` (recommended
    for RNA-seq when annotation matches the genome build).

    Intermediate ``*_ss.txt`` / ``*_exon.txt`` are written next to the index prefix parent
    unless ``work_dir`` is set.
    """
    ref_fasta = ref_fasta.resolve()
    index_prefix = index_prefix.resolve()
    if not ref_fasta.is_file():
        raise FileNotFoundError(ref_fasta)

    out_parent = index_prefix.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    staging = work_dir.resolve() if work_dir else out_parent
    staging.mkdir(parents=True, exist_ok=True)

    ss_path: Optional[Path] = None
    exon_path: Optional[Path] = None
    if ref_gtf is not None:
        ref_gtf = ref_gtf.resolve()
        if not ref_gtf.is_file():
            raise FileNotFoundError(ref_gtf)
        hdir = _hisat2_dir(cfg)
        ss_path = staging / f"{index_prefix.name}_splice_sites.txt"
        exon_path = staging / f"{index_prefix.name}_exons.txt"
        py = sys.executable
        for script, out in (
            ("extract_splice_sites.py", ss_path),
            ("extract_exons.py", exon_path),
        ):
            sp = hdir / script
            if not sp.is_file():
                raise FileNotFoundError(f"Missing HISAT2 helper script: {sp}")
            LOG.info("Running %s -> %s", script, out)
            with open(out, "w") as fout:
                subprocess.run(
                    [py, str(sp), str(ref_gtf)],
                    stdout=fout,
                    check=True,
                    cwd=str(staging),
                )

    cmd = [
        str(cfg.hisat2_build),
        "-p",
        str(threads),
    ]
    if ss_path is not None and exon_path is not None:
        cmd.extend(["--ss", str(ss_path), "--exon", str(exon_path)])
    cmd.extend([str(ref_fasta), str(index_prefix)])

    LOG.info("hisat2-build %s", " ".join(cmd))
    conda_run_cmd(cfg.conda_env, cmd, cwd=str(out_parent), check=True)

    # Verify index
    inspect_hisat2_index(cfg, index_prefix)
    return index_prefix


def inspect_hisat2_index(cfg: RnaseqToolConfig, index_prefix: Path) -> None:
    """Run ``hisat2-inspect -s`` on the index prefix."""
    conda_run_cmd(
        cfg.conda_env,
        [str(cfg.hisat2_inspect), "-s", str(index_prefix)],
        cwd=str(index_prefix.parent),
        check=True,
    )
