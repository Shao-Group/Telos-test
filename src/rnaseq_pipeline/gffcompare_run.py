"""GFFCompare vs reference GTF (``gffcmp/`` layout like the shell scripts)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from rnaseq_pipeline.config import RnaseqToolConfig
from rnaseq_pipeline.exec_env import conda_run_cmd

LOG = logging.getLogger(__name__)


def run_gffcompare_pair(
    query_gtf: Path,
    ref_gtf: Path,
    out_prefix: str,
    cfg: RnaseqToolConfig,
    *,
    gffcmp_dir: Path,
) -> None:
    """``gffcompare -r ref -o <prefix> query`` from inside ``gffcmp_dir``."""
    gffcmp_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        str(cfg.gffcompare),
        "-r",
        str(ref_gtf.resolve()),
        "-o",
        out_prefix,
        str(query_gtf.resolve()),
    ]
    conda_run_cmd(cfg.conda_env, cmd, cwd=gffcmp_dir, check=True)


def run_gffcompare_batch(
    pairs: Iterable[tuple[Path, str]],
    ref_gtf: Path,
    cfg: RnaseqToolConfig,
    *,
    work_dir: Path,
) -> Path:
    """
    For each ``(gtf_path, prefix)`` run gffcompare; outputs under ``work_dir/gffcmp/``.

    Returns the ``gffcmp`` directory path.
    """
    gffcmp_dir = work_dir / "gffcmp"
    gffcmp_dir.mkdir(parents=True, exist_ok=True)
    for gtf, prefix in pairs:
        LOG.info("gffcompare -o %s %s", prefix, gtf.name)
        run_gffcompare_pair(gtf, ref_gtf, prefix, cfg, gffcmp_dir=gffcmp_dir)
    return gffcmp_dir
