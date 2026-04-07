from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunIO:
    bam: Path
    gtf: Path
    outdir: Path


@dataclass(frozen=True)
class TrainIO(RunIO):
    ref_gtf: Path
    tmap: Path | None = None


@dataclass(frozen=True)
class PredictIO(RunIO):
    model_dir: Path


@dataclass(frozen=True)
class BenchmarkIO:
    config: Path
    outdir: Path
