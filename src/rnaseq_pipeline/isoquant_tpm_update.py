"""Update IsoQuant GTF transcript coverage values using TPM tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import tempfile


_TX_ID_RE = re.compile(r'(?<![A-Za-z0-9_])transcript_id "([^"]+)"')
_COVERAGE_RE = re.compile(r'(?<![A-Za-z0-9_])coverage "([^"]*)"')
_COV_RE = re.compile(r'(?<![A-Za-z0-9_])cov "([^"]*)"')


@dataclass(frozen=True)
class IsoquantUpdateResult:
    """Per-file update summary."""

    gtf: Path
    tpm: Path
    out_gtf: Path
    transcript_lines: int
    updated_from_tpm: int
    missing_in_tpm: int


def _parse_transcript_id(attrs: str) -> str | None:
    match = _TX_ID_RE.search(attrs)
    if not match:
        return None
    txid = match.group(1).strip()
    return txid or None


def _set_coverage_attr(attributes: str, value: float) -> str:
    cov_literal = f'coverage "{value:.6f}"'
    if _COVERAGE_RE.search(attributes):
        return _COVERAGE_RE.sub(cov_literal, attributes, count=1)
    if _COV_RE.search(attributes):
        return _COV_RE.sub(cov_literal, attributes, count=1)
    attrs = attributes.rstrip()
    if attrs and not attrs.endswith(";"):
        attrs = f"{attrs};"
    if attrs:
        return f"{attrs} {cov_literal};"
    return f"{cov_literal};"


def load_isoquant_tpm_table(tpm_path: Path) -> dict[str, float]:
    """
    Parse IsoQuant transcript TPM table to a transcript->TPM mapping.

    Expected format is a two-column TSV with optional comment header:
    ``#feature_id<TAB>TPM``.
    """
    tpm_map: dict[str, float] = {}
    with tpm_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            txid = parts[0].strip()
            if not txid:
                continue
            try:
                tpm_val = float(parts[1].strip())
            except ValueError:
                continue
            tpm_map[txid] = tpm_val
    return tpm_map


def update_isoquant_gtf_coverage_from_tpm(
    gtf_path: Path,
    tpm_path: Path,
    out_gtf_path: Path,
) -> IsoquantUpdateResult:
    """
    Rewrite transcript lines in ``gtf_path`` with TPM-backed ``coverage`` values.

    - If transcript_id exists in TPM table, write that TPM into ``coverage``.
    - If transcript_id is absent, line is left unchanged.
    """
    if not gtf_path.is_file():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")
    if not tpm_path.is_file():
        raise FileNotFoundError(f"TPM table not found: {tpm_path}")

    tpm_map = load_isoquant_tpm_table(tpm_path)
    out_gtf_path.parent.mkdir(parents=True, exist_ok=True)
    same_path = out_gtf_path.resolve() == gtf_path.resolve()

    transcript_lines = 0
    updated_from_tpm = 0
    missing_in_tpm = 0

    final_out = out_gtf_path
    temp_out: Path | None = None
    if same_path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(out_gtf_path.parent),
            delete=False,
            prefix=f"{out_gtf_path.name}.tmp.",
        ) as handle:
            temp_out = Path(handle.name)
        final_out = temp_out

    with gtf_path.open("r", encoding="utf-8", errors="replace") as src, final_out.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                dst.write(raw)
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                dst.write(raw)
                continue

            transcript_lines += 1
            txid = _parse_transcript_id(cols[8])
            if txid is None:
                dst.write(raw)
                continue
            tpm = tpm_map.get(txid)
            if tpm is None:
                missing_in_tpm += 1
                dst.write(raw)
                continue

            cols[8] = _set_coverage_attr(cols[8], tpm)
            dst.write("\t".join(cols) + "\n")
            updated_from_tpm += 1

    if same_path and temp_out is not None:
        temp_out.replace(out_gtf_path)

    return IsoquantUpdateResult(
        gtf=gtf_path,
        tpm=tpm_path,
        out_gtf=out_gtf_path,
        transcript_lines=transcript_lines,
        updated_from_tpm=updated_from_tpm,
        missing_in_tpm=missing_in_tpm,
    )


def discover_isoquant_pairs(root: Path) -> list[tuple[Path, Path]]:
    """
    Discover ``(isoquant.gtf, isoquant_transcript_model_tpm.tsv)`` pairs under ``root``.
    """
    pairs: list[tuple[Path, Path]] = []
    for gtf in sorted(root.glob("**/isoquant.gtf")):
        tpm = gtf.with_name("isoquant_transcript_model_tpm.tsv")
        if tpm.is_file():
            pairs.append((gtf, tpm))
    return pairs
