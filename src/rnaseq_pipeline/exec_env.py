"""Run subprocess commands inside a Conda environment."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

LOG = logging.getLogger(__name__)

StrPath = Union[str, Path]


def conda_run_cmd(
    conda_env: str,
    inner: Sequence[str],
    *,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run ``inner`` as: ``conda run -n <env> --no-capture-output <inner...>``.

    ``--no-capture-output`` streams stdout/stderr for long jobs (IsoQuant, etc.).
    """
    cmd: List[str] = [
        "conda",
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
        *[str(x) for x in inner],
    ]
    LOG.info("Running: %s (cwd=%s)", " ".join(cmd), cwd or os.getcwd())
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        check=check,
    )
