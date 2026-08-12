"""Config loading: product telos helpers + repo Stage I defaults path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from telos.config_loader import get_nested, load_mapping_config

__all__ = ["default_stage1_config_path", "get_nested", "load_mapping_config"]


def default_stage1_config_path() -> Path:
    """Repo-root ``configs/stage1.defaults.yaml``."""
    return Path(__file__).resolve().parents[2] / "configs" / "stage1.defaults.yaml"
