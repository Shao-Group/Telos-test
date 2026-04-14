from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


class GtfcuffError(RuntimeError):
    pass


def default_gtfcuff_search_paths() -> list[Path]:
    env = os.environ.get("TELOS_GTFCUFF", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend(
        [
            Path("/datadisk1/ixk5174/tools/rnaseqtools/gtfcuff/gtfcuff"),
            Path(__file__).resolve().parents[3] / "tools" / "rnaseqtools" / "gtfcuff" / "gtfcuff",
        ]
    )
    return paths


def resolve_gtfcuff_binary(explicit: str | Path | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        raise GtfcuffError(f"gtfcuff binary not found or not executable: {p}")
    which = shutil.which("gtfcuff")
    if which:
        return Path(which).resolve()
    for candidate in default_gtfcuff_search_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise GtfcuffError(
        "gtfcuff not found. Set TELOS_GTFCUFF or config analysis.pr_vs_baseline.gtfcuff_bin, "
        "or install rnaseqtools and ensure gtfcuff is on PATH."
    )


def run_gtfcuff_roc(
    gtfcuff_bin: Path,
    tmap_path: Path,
    ref_multi_exon_count: int,
    measure: str = "cov",
) -> str:
    """Run ``gtfcuff roc <tmap> <ref-size> <cov|TPM|FPKM>``; return stdout text."""
    proc = subprocess.run(
        [
            str(gtfcuff_bin),
            "roc",
            str(tmap_path.resolve()),
            str(int(ref_multi_exon_count)),
            measure,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfcuffError(
            f"gtfcuff roc failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def run_gtfcuff_auc(gtfcuff_bin: Path, tmap_path: Path, ref_multi_exon_count: int) -> float:
    """Parse ``auc =`` from ``gtfcuff auc`` stdout (legacy generate_roc_data.get_aupr)."""
    proc = subprocess.run(
        [
            str(gtfcuff_bin),
            "auc",
            str(tmap_path.resolve()),
            str(int(ref_multi_exon_count)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfcuffError(
            f"gtfcuff auc failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    match = re.search(r"auc\s*=\s*(\d+\.?\d*)", proc.stdout)
    if not match:
        raise GtfcuffError(f"Failed to parse auc from gtfcuff auc output: {proc.stdout!r}")
    return float(match.group(1))
