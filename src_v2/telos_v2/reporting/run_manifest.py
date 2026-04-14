from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telos_v2 import __version__


def _safe_git_hash(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def build_run_manifest(command: str, args_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "tool": "telos_v2",
        "tool_version": __version__,
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "env": {
            "cwd": os.getcwd(),
            "git_commit": _safe_git_hash(Path.cwd()),
        },
        "args": args_dict,
    }


def write_run_manifest(manifest: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out
