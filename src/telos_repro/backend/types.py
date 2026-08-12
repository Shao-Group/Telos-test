"""Neutral train/predict request types for the Track B facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_path(value: Any, field: str) -> Path:
    if value is None:
        raise ValueError(f"{field} is required")
    return value if isinstance(value, Path) else Path(str(value))


def _as_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _as_optional_path_tuple(value: Any) -> tuple[Path, ...] | None:
    if value is None:
        return None
    return tuple(_as_path(p, "pool_entry") for p in value)


@dataclass(frozen=True, kw_only=True)
class TrainRequest:
    bam: Path
    gtf: Path
    outdir: Path
    ref_gtf: Path
    tmap: Path | None = None
    config_file: Path | None = None
    gtf_pool: tuple[Path, ...] | None = None
    tmap_pool: tuple[Path, ...] | None = None
    stage1_no_parallel: bool = False
    stage1_n_workers: int | None = None
    split_policy: str | None = None
    n_jobs: int | None = None


@dataclass(frozen=True, kw_only=True)
class PredictRequest:
    bam: Path
    gtf: Path
    outdir: Path
    model_dir: Path
    config_file: Path | None = None
    stage1_no_parallel: bool = False
    stage1_n_workers: int | None = None
    backend: str = "xgb"
    min_score: float | None = None


def as_train_request(io: object) -> TrainRequest:
    """Accept ``TrainRequest``, ``telos_repro``/``telos`` ``TrainIO``, or a duck-typed object."""
    if isinstance(io, TrainRequest):
        return io
    return TrainRequest(
        bam=_as_path(getattr(io, "bam"), "bam"),
        gtf=_as_path(getattr(io, "gtf"), "gtf"),
        outdir=_as_path(getattr(io, "outdir"), "outdir"),
        ref_gtf=_as_path(getattr(io, "ref_gtf"), "ref_gtf"),
        tmap=_as_optional_path(getattr(io, "tmap", None)),
        config_file=_as_optional_path(getattr(io, "config_file", None)),
        gtf_pool=_as_optional_path_tuple(getattr(io, "gtf_pool", None)),
        tmap_pool=_as_optional_path_tuple(getattr(io, "tmap_pool", None)),
        stage1_no_parallel=bool(getattr(io, "stage1_no_parallel", False)),
        stage1_n_workers=getattr(io, "stage1_n_workers", None),
        split_policy=getattr(io, "split_policy", None),
        n_jobs=getattr(io, "n_jobs", None),
    )


def as_predict_request(io: object) -> PredictRequest:
    if isinstance(io, PredictRequest):
        return io
    return PredictRequest(
        bam=_as_path(getattr(io, "bam"), "bam"),
        gtf=_as_path(getattr(io, "gtf"), "gtf"),
        outdir=_as_path(getattr(io, "outdir"), "outdir"),
        model_dir=_as_path(getattr(io, "model_dir"), "model_dir"),
        config_file=_as_optional_path(getattr(io, "config_file", None)),
        stage1_no_parallel=bool(getattr(io, "stage1_no_parallel", False)),
        stage1_n_workers=getattr(io, "stage1_n_workers", None),
        backend=str(getattr(io, "backend", "xgb") or "xgb"),
        min_score=getattr(io, "min_score", None),
    )
