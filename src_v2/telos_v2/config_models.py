from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class RunIO:
    bam: Path
    gtf: Path
    outdir: Path
    config_file: Path | None = None


@dataclass(frozen=True, kw_only=True)
class TrainIO(RunIO):
    ref_gtf: Path
    tmap: Path | None = None
    stage1_no_parallel: bool = False
    stage1_n_workers: int | None = None


@dataclass(frozen=True, kw_only=True)
class PredictIO(RunIO):
    model_dir: Path
    stage1_no_parallel: bool = False
    stage1_n_workers: int | None = None


@dataclass(frozen=True)
class BenchmarkIO:
    config: Path
    outdir: Path


@dataclass(frozen=True, kw_only=True)
class FilterGtfIO:
    """Inputs for gtfformat remove-fp / remove-fp-threshold."""

    gtf: Path
    predictions: Path
    out_gtf: Path
    hard_mode: int = 0
    mode: str = "threshold"  # "exact" -> remove-fp; "threshold" -> remove-fp-threshold
    bp_threshold: int = 0
    gtfformat_bin: str | Path | None = None
    config_file: Path | None = None
