"""
Shared training directories for cross-annotation benchmark grids.

Train once per ``(data_type, train_annotation)`` under ``<outdir>/<subdir>/<dt>__train_<anno>/``;
matrix cells with the same train axis reuse ``train.mode=skip`` when model artifacts exist.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from telos_v2.models import (
    STAGE1_BACKENDS,
    stage1_bundle_path,
    stage2_feature_names_json_for_backend,
    stage2_model_joblib_for_backend,
)


def shared_train_dir(
    outdir: Path,
    data_type: str,
    train_annotation: str,
    *,
    subdir: str = "_shared_train",
) -> Path:
    """Directory for one train run shared across all test annotations for this train axis."""
    return (outdir / subdir / f"{data_type}__train_{train_annotation}").resolve()


def shared_models_ready(model_dir: Path) -> bool:
    """True when all Stage I bundles and Stage II joblibs + feature JSONs exist under ``model_dir``."""
    if not model_dir.is_dir():
        return False
    for site_type in ("TSS", "TES"):
        for backend in STAGE1_BACKENDS:
            if not (model_dir / stage1_bundle_path(site_type, backend)).is_file():
                return False
    for backend in STAGE1_BACKENDS:
        if not (model_dir / stage2_model_joblib_for_backend(backend)).is_file():
            return False
        if not (model_dir / stage2_feature_names_json_for_backend(backend)).is_file():
            return False
    return True


@contextmanager
def shared_train_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive lock for one shared-train directory (safe across processes)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def apply_train_reuse(mapping: dict[str, Any], *, shared_train_parent: Path) -> None:
    """
    Point ``train.outdir`` at ``shared_train_parent``; skip training when models already exist.

    Sets ``train.mode=skip`` and ``train.model_dir`` when ``shared_train_parent/models`` is complete;
    otherwise ``train.mode=run``.
    """
    train_block = mapping.get("train")
    if not isinstance(train_block, dict):
        raise ValueError("mapping.train must be a dict")
    train_block["outdir"] = str(shared_train_parent)
    md = shared_train_parent / "models"
    md_resolved = md.resolve()
    if shared_models_ready(md_resolved):
        train_block["mode"] = "skip"
        train_block["model_dir"] = str(md_resolved)
    else:
        train_block["mode"] = "run"
        train_block.pop("model_dir", None)


__all__ = [
    "apply_train_reuse",
    "shared_models_ready",
    "shared_train_dir",
    "shared_train_lock",
]
