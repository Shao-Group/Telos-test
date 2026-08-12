"""Resolve repo root and load ``configs/paths.yaml`` (or the example stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PATH_KEYS = (
    "frozen_root",
    "repo_root",
    "bundles_root",
    "genome_root",
    "fastq_root",
    "goldens_runs_root",
    "goldens_figures_root",
    "runs_root",
    "figures_root",
    "gffcompare_bin",
    "telos_stage1_cache_dir",
    "telos_checkout",
    "telos_pin_sha",
    "backend",
)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents until ``pyproject.toml`` + ``src/telos_repro`` are found."""
    cur = (start or Path.cwd()).resolve()
    for base in (cur, *cur.parents):
        if (base / "pyproject.toml").is_file() and (base / "src" / "telos_repro").is_dir():
            return base
    raise FileNotFoundError(
        "Could not locate Telos-repro root (expected pyproject.toml + src/telos_repro). "
        "Run from the clone or set cwd accordingly."
    )


def paths_config_file(repo_root: Path) -> Path:
    local = repo_root / "configs" / "paths.yaml"
    if local.is_file():
        return local
    example = repo_root / "configs" / "paths.example.yaml"
    if example.is_file():
        return example
    raise FileNotFoundError(
        f"Missing {local} and {example}. Copy paths.example.yaml to paths.yaml."
    )


def load_paths(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    cfg_path = paths_config_file(root)
    with cfg_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {cfg_path}")

    out: dict[str, Any] = {"_config_path": str(cfg_path), "repo_root": str(root)}
    for key in _PATH_KEYS:
        if key in raw and raw[key] is not None:
            out[key] = str(raw[key])
    out.setdefault("repo_root", str(root))
    out.setdefault("runs_root", str(root / "runs"))
    out.setdefault("figures_root", str(root / "plot_v2"))
    return out


def path_value(paths: dict[str, Any], key: str) -> Path:
    """Return a path from ``paths``. Relative entries resolve against ``repo_root``.

    ``gffcompare_bin`` may be a bare command name (e.g. ``gffcompare``); those are
    left unresolved so callers can put them on ``PATH`` / ``GFFCOMPARE``.
    """
    if key not in paths or not paths[key]:
        raise KeyError(f"paths config missing {key!r}")
    raw = str(paths[key]).strip()
    p = Path(raw).expanduser()
    if key == "gffcompare_bin" and not p.is_absolute() and len(p.parts) == 1:
        return p
    if not p.is_absolute():
        root = Path(str(paths.get("repo_root", "."))).expanduser()
        p = (root / p).resolve()
    return p
