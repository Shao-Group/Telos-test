from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class GtfformatError(RuntimeError):
    pass


def default_gtfformat_search_paths() -> list[Path]:
    env = os.environ.get("TELOS_GTFFORMAT", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend(
        [
            Path("/datadisk1/ixk5174/tools/rnaseqtools/gtfformat/gtfformat"),
            Path(__file__).resolve().parents[3] / "tools" / "rnaseqtools" / "gtfformat" / "gtfformat",
        ]
    )
    return paths


def resolve_gtfformat_binary(explicit: str | Path | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        raise GtfformatError(f"gtfformat binary not found or not executable: {p}")
    which = shutil.which("gtfformat")
    if which:
        return Path(which).resolve()
    for candidate in default_gtfformat_search_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise GtfformatError(
        "gtfformat not found. Set TELOS_GTFFORMAT or config rnaseqtools.gtfformat_bin, "
        "or install rnaseqtools and ensure gtfformat is on PATH."
    )


def run_tsstes(gtfformat_bin: Path, gtf: Path) -> str:
    gtf = gtf.resolve()
    if not gtf.is_file():
        raise GtfformatError(f"GTF not found: {gtf}")
    try:
        proc = subprocess.run(
            [str(gtfformat_bin), "TSSTES", str(gtf)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GtfformatError(f"Failed to run gtfformat TSSTES: {exc}") from exc
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat TSSTES failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def run_get_cov(gtfformat_bin: Path, gtf: Path, out_tsv: Path) -> None:
    gtf = gtf.resolve()
    out_tsv = out_tsv.resolve()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not gtf.is_file():
        raise GtfformatError(f"GTF not found: {gtf}")
    try:
        proc = subprocess.run(
            [str(gtfformat_bin), "get-cov", str(gtf), str(out_tsv)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GtfformatError(f"Failed to run gtfformat get-cov: {exc}") from exc
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat get-cov failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )


def run_remove_fp(
    gtfformat_bin: Path,
    in_gtf: Path,
    predictions_tsv: Path,
    out_gtf: Path,
    hard_mode: int,
) -> None:
    in_gtf, predictions_tsv, out_gtf = in_gtf.resolve(), predictions_tsv.resolve(), out_gtf.resolve()
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(gtfformat_bin),
            "remove-fp",
            str(in_gtf),
            str(predictions_tsv),
            str(out_gtf),
            str(int(hard_mode)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat remove-fp failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )


def run_remove_fp_threshold(
    gtfformat_bin: Path,
    in_gtf: Path,
    predictions_tsv: Path,
    out_gtf: Path,
    hard_mode: int,
    bp_threshold: int,
) -> None:
    in_gtf, predictions_tsv, out_gtf = in_gtf.resolve(), predictions_tsv.resolve(), out_gtf.resolve()
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(gtfformat_bin),
            "remove-fp-threshold",
            str(in_gtf),
            str(predictions_tsv),
            str(out_gtf),
            str(int(hard_mode)),
            str(int(bp_threshold)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat remove-fp-threshold failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def run_filter_chrom(
    gtfformat_bin: Path,
    in_gtf: Path,
    chrom_list_file: Path,
    out_gtf: Path,
) -> None:
    proc = subprocess.run(
        [
            str(gtfformat_bin),
            "filter-chrom",
            str(in_gtf.resolve()),
            str(chrom_list_file.resolve()),
            str(out_gtf.resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat filter-chrom failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def run_update_transcript_cov(
    gtfformat_bin: Path,
    in_gtf: Path,
    predictions_tsv: Path,
    out_gtf: Path,
) -> None:
    proc = subprocess.run(
        [
            str(gtfformat_bin),
            "update-transcript-cov",
            str(in_gtf.resolve()),
            str(predictions_tsv.resolve()),
            str(out_gtf.resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GtfformatError(
            f"gtfformat update-transcript-cov failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
