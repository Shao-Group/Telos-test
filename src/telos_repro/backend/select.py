"""Backend selection and Protocol.

Train/predict always go through pinned product ``telos`` after vendored core deletion.
"""

from __future__ import annotations

import os
from typing import Protocol

from telos_repro.backend.types import PredictRequest, TrainRequest
from telos_repro.paths import find_repo_root, load_paths


class TrainPredictBackend(Protocol):
    name: str

    def train(self, req: TrainRequest) -> int: ...

    def predict(self, req: PredictRequest) -> int: ...


_VALID = ("telos",)


def get_backend_name(*, explicit: str | None = None) -> str:
    if explicit:
        name = explicit.strip()
    else:
        env = os.environ.get("TELOS_REPRO_BACKEND", "").strip()
        if env:
            name = env
        else:
            try:
                paths = load_paths(find_repo_root())
                name = str(paths.get("backend") or "telos").strip()
            except FileNotFoundError:
                name = "telos"
    if name == "telos_v2":
        raise ValueError(
            "backend 'telos_v2' was removed after Track B core deletion; "
            "use TELOS_REPRO_BACKEND=telos (pinned product Telos)."
        )
    if name not in _VALID:
        raise ValueError(f"Unknown backend {name!r}; expected one of {_VALID}")
    return name


def resolve_backend(*, explicit: str | None = None) -> TrainPredictBackend:
    get_backend_name(explicit=explicit)  # validate
    from telos_repro.backend.telos_backend import TelosBackend

    return TelosBackend()
