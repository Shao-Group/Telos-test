"""Benchmark summary columns derived from training artifacts (no CLI / pysam imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def train_validation_aupr_columns(reports_dir: Path) -> dict[str, Any]:
    """
    Validation-split AUPR from training (Stage I per site-type × backend, Stage II per Stage I backend).
    Written by :func:`telos_v2.models.stage1_train.write_train_metrics`.
    """
    path = reports_dir / "train_metrics.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for st in ("tss", "tes"):
        for b in ("rf", "xgb"):
            block = data.get(f"{st}_{b}")
            if isinstance(block, dict):
                v = block.get("aupr")
                if v is not None:
                    out[f"train_val_aupr_stage1_{st}_{b}"] = float(v)
    for b in ("rf", "xgb"):
        block = data.get(f"stage2_{b}")
        if isinstance(block, dict):
            v = block.get("aupr")
            if v is not None:
                out[f"train_val_aupr_stage2_{b}"] = float(v)
    return out
