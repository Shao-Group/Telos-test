"""Train/predict backend facade for Track B.

Call sites in ``telos_repro.benchmark`` / CLI should use::

    from telos_repro.backend import train, predict

Backend:
- ``telos`` — pinned product package (``telos_checkout``)

The vendored ``telos_v2`` train/predict stack was removed after Tier-1 golden parity.
Select via ``TELOS_REPRO_BACKEND`` or ``configs/paths.yaml`` key ``backend`` (must be ``telos``).
"""

from __future__ import annotations

from telos_repro.backend.select import get_backend_name, resolve_backend
from telos_repro.backend.types import PredictRequest, TrainRequest, as_predict_request, as_train_request

__all__ = [
    "PredictRequest",
    "TrainRequest",
    "as_predict_request",
    "as_train_request",
    "get_backend_name",
    "predict",
    "resolve_backend",
    "train",
]


def train(io: object) -> int:
    """Run Stage I+II training via the selected backend."""
    return resolve_backend().train(as_train_request(io))


def predict(io: object) -> int:
    """Run Stage I+II prediction via the selected backend."""
    return resolve_backend().predict(as_predict_request(io))
