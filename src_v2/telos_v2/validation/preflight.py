from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class PreflightError(RuntimeError):
    """Raised when required inputs fail validation."""


@dataclass(frozen=True)
class RunLayout:
    root: Path
    models_dir: Path
    predictions_dir: Path
    filtered_dir: Path
    reports_dir: Path
    debug_dir: Path


def ensure_run_layout(outdir: Path, save_intermediates: bool = False) -> RunLayout:
    root = outdir.resolve()
    models_dir = root / "models"
    predictions_dir = root / "predictions"
    filtered_dir = root / "filtered"
    reports_dir = root / "reports"
    debug_dir = root / "debug"

    for d in (models_dir, predictions_dir, filtered_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    if save_intermediates:
        debug_dir.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        root=root,
        models_dir=models_dir,
        predictions_dir=predictions_dir,
        filtered_dir=filtered_dir,
        reports_dir=reports_dir,
        debug_dir=debug_dir,
    )


def _validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise PreflightError(f"{label} not found: {path}")
    if not path.is_file():
        raise PreflightError(f"{label} must be a file: {path}")


def validate_input_file(path: Path, label: str) -> None:
    _validate_file_exists(path, label)


def validate_input_gtf(gtf_path: Path) -> None:
    _validate_gtf_format(gtf_path)


def _validate_gtf_format(gtf_path: Path) -> None:
    """
    Lightweight GTF sanity check:
    - file exists
    - at least one non-comment record
    - first non-comment record has >= 9 tab-separated fields
    """
    _validate_file_exists(gtf_path, "GTF")
    non_comment_seen = False
    with gtf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            non_comment_seen = True
            cols = line.split("\t")
            if len(cols) < 9:
                raise PreflightError(
                    f"GTF appears malformed (expected >=9 columns): {gtf_path}"
                )
            break
    if not non_comment_seen:
        raise PreflightError(f"GTF has no feature records: {gtf_path}")


def _candidate_bam_indices(bam_path: Path) -> Iterable[Path]:
    # Common index naming variants.
    yield bam_path.with_suffix(bam_path.suffix + ".bai")  # sample.bam.bai
    yield bam_path.with_suffix(".bai")  # sample.bai
    yield bam_path.with_suffix(bam_path.suffix + ".csi")  # sample.bam.csi
    yield bam_path.with_suffix(".csi")  # sample.csi


def _validate_bam_and_index(bam_path: Path) -> None:
    _validate_file_exists(bam_path, "BAM")
    if bam_path.suffix.lower() != ".bam":
        raise PreflightError(f"Expected a .bam input: {bam_path}")

    has_index = any(idx.exists() and idx.is_file() for idx in _candidate_bam_indices(bam_path))
    if not has_index:
        raise PreflightError(
            f"BAM index not found for {bam_path}. Expected one of .bai/.csi variants."
        )

    # Try to verify coordinate sort order via pysam header.
    try:
        import pysam  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise PreflightError(
            "pysam is required for BAM validation but is not available."
        ) from exc

    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            hd = dict(bam.header).get("HD", {})
            so = hd.get("SO", "")
            if str(so).lower() != "coordinate":
                raise PreflightError(
                    f"BAM must be coordinate-sorted (HD:SO=coordinate), got SO={so!r} for {bam_path}"
                )
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError(f"Unable to open/validate BAM: {bam_path}") from exc


def _preflight_gtfformat_if_needed(cfg_map: dict[str, Any] | None) -> None:
    if not cfg_map:
        cfg_map = {}
    from telos_v2.backends.gtfformat import GtfformatError
    from telos_v2.candidates.load import gtfformat_needed_for_run, resolve_gtfformat_for_config

    if not gtfformat_needed_for_run(cfg_map):
        return
    try:
        resolve_gtfformat_for_config(cfg_map)
    except GtfformatError as exc:
        raise PreflightError(str(exc)) from exc


def run_preflight_train(
    bam: Path,
    gtf: Path,
    ref_gtf: Path,
    tmap: Path | None = None,
    cfg_map: dict[str, Any] | None = None,
) -> None:
    _validate_bam_and_index(bam)
    _validate_gtf_format(gtf)
    _validate_gtf_format(ref_gtf)
    if tmap is None:
        raise PreflightError(
            "--tmap is required for training (Stage II transcript labels from gffcompare .tmap)."
        )
    _validate_file_exists(tmap, "TMAP")
    _preflight_gtfformat_if_needed(cfg_map)


def run_preflight_predict(
    bam: Path,
    gtf: Path,
    model_dir: Path,
    cfg_map: dict[str, Any] | None = None,
) -> None:
    from telos_v2.models import (
        STAGE1_BACKENDS,
        stage1_bundle_path,
        stage2_feature_names_json_for_backend,
        stage2_model_joblib_for_backend,
    )

    _validate_bam_and_index(bam)
    _validate_gtf_format(gtf)
    if not model_dir.exists() or not model_dir.is_dir():
        raise PreflightError(f"Model directory not found: {model_dir}")
    for site_type in ("TSS", "TES"):
        for backend in STAGE1_BACKENDS:
            name = stage1_bundle_path(site_type, backend)
            p = model_dir / name
            if not p.is_file():
                raise PreflightError(f"Missing Stage I model artifact: {p}")
    for backend in STAGE1_BACKENDS:
        m = model_dir / stage2_model_joblib_for_backend(backend)
        if not m.is_file():
            raise PreflightError(f"Missing Stage II model artifact: {m}")
        j = model_dir / stage2_feature_names_json_for_backend(backend)
        if not j.is_file():
            raise PreflightError(f"Missing Stage II feature list: {j}")
    _preflight_gtfformat_if_needed(cfg_map)


def run_preflight_benchmark(config: Path) -> None:
    _validate_file_exists(config, "Benchmark config")
    # Lightweight sanity: must be parseable as YAML-like JSON subset.
    # Full YAML schema validation will be added in a later phase.
    txt = config.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        raise PreflightError(f"Benchmark config is empty: {config}")
    if txt.startswith("{"):
        try:
            json.loads(txt)
        except Exception as exc:
            raise PreflightError(
                f"Benchmark config appears to be JSON but is invalid: {config}"
            ) from exc
