"""Discover FASTQ inputs from a file or directory."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastqInput:
    """Single-end or paired FASTQ paths (plain or gz)."""

    r1: List[Path]
    r2: Optional[List[Path]] = None

    @property
    def is_paired(self) -> bool:
        return self.r2 is not None and len(self.r2) > 0


def _is_fastq(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz"))


def _sort_key(p: Path) -> str:
    return p.name


def discover_fastq(path: Path) -> FastqInput:
    """
    If ``path`` is a file, treat as single-end (one sample).

    If ``path`` is a directory, collect ``*.fastq`` / ``*.fq`` / ``*.gz`` variants.
    Pairing heuristic: names containing ``_1`` / ``_2`` or ``.R1`` / ``.R2`` (case-insensitive).
    """
    path = path.resolve()
    if path.is_file():
        if not _is_fastq(path):
            raise ValueError(f"Not a FASTQ file: {path}")
        return FastqInput(r1=[path], r2=None)

    if not path.is_dir():
        raise ValueError(f"FASTQ path not found: {path}")

    files = sorted(
        [p for p in path.iterdir() if p.is_file() and _is_fastq(p)],
        key=_sort_key,
    )
    if not files:
        raise ValueError(f"No FASTQ files in directory: {path}")

    pairs_r1: List[Path] = []
    pairs_r2: List[Path] = []
    singles: List[Path] = []

    pat_1 = re.compile(r"([._-])R?1(\.|_|$)", re.IGNORECASE)
    pat_2 = re.compile(r"([._-])R?2(\.|_|$)", re.IGNORECASE)

    for f in files:
        n = f.name
        if pat_2.search(n):
            pairs_r2.append(f)
        elif pat_1.search(n):
            pairs_r1.append(f)
        else:
            singles.append(f)

    if pairs_r1 or pairs_r2:
        if len(pairs_r1) != len(pairs_r2):
            raise ValueError(
                f"Mismatched R1/R2 counts in {path}: r1={len(pairs_r1)} r2={len(pairs_r2)}"
            )
        # Sort by basename stem so R1/R2 align
        def stem(p: Path) -> str:
            s = p.name
            for suf in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
                if s.lower().endswith(suf):
                    return s[: -len(suf)]
            return s

        pairs_r1 = sorted(pairs_r1, key=stem)
        pairs_r2 = sorted(pairs_r2, key=stem)
        for a, b in zip(pairs_r1, pairs_r2):
            if stem(a).replace("1", "2") != stem(b) and stem(a).replace("_1", "_2") != stem(b):
                LOG.warning("R1/R2 stem pairing uncertain: %s vs %s", a.name, b.name)
        return FastqInput(r1=pairs_r1, r2=pairs_r2)

    return FastqInput(r1=singles, r2=None)
