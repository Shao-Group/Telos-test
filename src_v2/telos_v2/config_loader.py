from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mapping_config(path: Path | None) -> dict[str, Any]:
    """
    Load a run config mapping from YAML or JSON.
    Returns empty dict when path is None.
    """
    if path is None:
        return {}
    if not path.exists() or not path.is_file():
        raise ValueError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level config must be a mapping/object.")
        return data

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError(
                "YAML config requested but PyYAML is not installed. "
                "Install pyyaml or use a JSON config."
            ) from exc
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level config must be a mapping/object.")
        return data

    raise ValueError(f"Unsupported config extension: {path.suffix}")


def get_nested(cfg: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
