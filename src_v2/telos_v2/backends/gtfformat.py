"""
Pure-Python and optional-binary **gtfformat** operations on GTF.

Includes transcript coverage tables, TSSTES counting, chromosome filtering, score injection into
transcript ``coverage`` attributes, and subprocess helpers for ``get-cov`` when a binary is available.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from telos_v2.gtf_attributes import parse_transcript_id


class GtfformatError(RuntimeError):
    """Raised for missing inputs, failed subprocess calls, or invalid GTF operations in this module."""


_COVERAGE_RE = re.compile(r'(?<![A-Za-z0-9_])coverage "([^"]*)"')
_COV_RE = re.compile(r'(?<![A-Za-z0-9_])cov "([^"]*)"')


def _replace_or_append_coverage(attributes: str, value: float) -> str:
    """
    Set ``coverage "<value>"`` in a GTF attribute string.

    If ``coverage "..."`` or ``cov "..."`` exists, replace the first match; otherwise append
    ``coverage "<value>";`` with proper semicolon separation.
    """
    # Prefer legacy/StringTie-compatible key "cov" for downstream gffcompare/tmap pipelines.
    cov_literal = f'cov "{value}"'
    if _COV_RE.search(attributes):
        return _COV_RE.sub(cov_literal, attributes, count=1)
    if _COVERAGE_RE.search(attributes):
        return _COVERAGE_RE.sub(cov_literal, attributes, count=1)
    attrs = attributes.rstrip()
    if attrs and not attrs.endswith(";"):
        attrs = f"{attrs};"
    if attrs:
        return f"{attrs} {cov_literal};"
    return f"{cov_literal};"


def _parse_coverage(attributes: str) -> float:
    """Read numeric ``coverage`` or ``cov`` attribute; return ``0.0`` if absent or unparsable."""
    match = _COVERAGE_RE.search(attributes) or _COV_RE.search(attributes)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def build_cov_dataframe(gtf: Path) -> pd.DataFrame:
    """
    One row per **transcript** with TSS/TES coordinates, coverage, and exon statistics.

    Parses transcript and exon lines; aggregates exon lengths per ``transcript_id``. TSS/TES positions
    follow strand: ``+`` uses ``start+1`` / ``end``, ``-`` uses ``end`` / ``start+1``.

    Raises:
        GtfformatError: If ``gtf`` is not a file.

    Returns:
        DataFrame with columns including ``transcript_id``, ``strand``, ``tss_chrom``, ``tes_chrom``,
        ``tss_pos``, ``tes_pos``, ``coverage``, exon count/length summaries.
    """
    gtf = gtf.resolve()
    if not gtf.is_file():
        raise GtfformatError(f"GTF not found: {gtf}")

    # Transcript coordinate summary rows.
    # For a given transcript_id we usually keep one row (strand '+' or '-').
    # When the GTF strand is '.' we emit two rows (one for '+' convention and one for '-' convention)
    # so downstream joins never silently drop ambiguous-strand transcripts.
    tx_rows: dict[str, list[dict[str, object]]] = {}
    tx_exons: dict[str, list[tuple[int, int]]] = {}

    with gtf.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            chrom, feature, start_s, end_s, strand, attrs = (
                cols[0],
                cols[2],
                cols[3],
                cols[4],
                cols[6],
                cols[8],
            )
            txid = parse_transcript_id(attrs)
            if not txid:
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if feature == "transcript":
                cov_val = _parse_coverage(attrs)
                if strand in {"+", "-"}:
                    canon = strand
                    tss_pos = start + 1 if canon == "+" else end
                    tes_pos = end if canon == "+" else start + 1
                    tx_rows.setdefault(txid, []).append(
                        {
                            "transcript_id": txid,
                            "strand": canon,
                            "tss_chrom": chrom,
                            "tes_chrom": chrom,
                            "tss_pos": tss_pos,
                            "tes_pos": tes_pos,
                            "coverage": cov_val,
                        }
                    )
                elif strand == ".":
                    # Ambiguous strand: emit both '+' and '-' coordinate conventions.
                    # This is a data-augmentation style choice to avoid losing those transcripts.
                    for canon in ("+", "-"):
                        tss_pos = start + 1 if canon == "+" else end
                        tes_pos = end if canon == "+" else start + 1
                        tx_rows.setdefault(txid, []).append(
                            {
                                "transcript_id": txid,
                                "strand": canon,
                                "tss_chrom": chrom,
                                "tes_chrom": chrom,
                                "tss_pos": tss_pos,
                                "tes_pos": tes_pos,
                                "coverage": cov_val,
                            }
                        )
                else:
                    # Unknown strand symbol: skip.
                    continue
            elif feature == "exon":
                tx_exons.setdefault(txid, []).append((start, end))

    rows: list[dict[str, object]] = []
    for txid, row_variants in tx_rows.items():
        exons = tx_exons.get(txid, [])
        lengths = [max(0, e - s + 1) for s, e in exons]

        if exons:
            by_start = sorted(exons, key=lambda t: t[0])
            exon_count = len(lengths)
            total_len = int(sum(lengths))
            max_len = int(max(lengths))
            min_len = int(min(lengths))
            mean_len = float(total_len / exon_count) if exon_count else 0.0
        else:
            by_start = []
            exon_count = 0
            total_len = 0
            max_len = 0
            min_len = 0
            mean_len = 0.0

        for row in row_variants:
            if by_start:
                strand = str(row["strand"])
                first_exon = by_start[0] if strand == "+" else by_start[-1]
                last_exon = by_start[-1] if strand == "+" else by_start[0]
                first_len = max(0, first_exon[1] - first_exon[0] + 1)
                last_len = max(0, last_exon[1] - last_exon[0] + 1)
            else:
                first_len = 0
                last_len = 0

            rows.append(
                {
                    **row,
                    "first_exon_length": first_len,
                    "last_exon_length": last_len,
                    "max_exon_length": max_len,
                    "min_exon_length": min_len,
                    "mean_exon_length": mean_len,
                    "total_exon_length": total_len,
                    "exon_count": exon_count,
                }
            )

    return pd.DataFrame(rows)


def default_gtfformat_search_paths() -> list[Path]:
    """
    Candidate locations for a ``gtfformat`` executable: ``TELOS_GTFFORMAT`` then repo ``tools/`` path.

    Used only when resolving a binary for subprocess helpers; pure-Python paths ignore this.
    """
    env = os.environ.get("TELOS_GTFFORMAT", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    # Optional: checkout-relative path (no machine-specific absolute paths).
    paths.append(
        Path(__file__).resolve().parents[3] / "tools" / "rnaseqtools" / "gtfformat" / "gtfformat"
    )
    return paths


def resolve_gtfformat_binary(explicit: str | Path | None) -> Path:
    """
    Resolve an executable ``gtfformat`` path: explicit arg, ``PATH``, or :func:`default_gtfformat_search_paths`.

    Raises:
        GtfformatError: If no usable executable is found.
    """
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


def run_tsstes(gtfformat_bin: Path | None, gtf: Path) -> str:
    """
    Emit TSSTES-style lines for ``gtf`` without calling an external binary (``gtfformat_bin`` ignored).

    Counts per-strand transcript ends at TSS/TES coordinates (same convention as C++ TSSTES). Output
    lines: ``TSS <chrom> <pos> <plus_count> <minus_count>`` and similarly for TES, sorted by chrom/pos.
    """
    _ = gtfformat_bin
    gtf = gtf.resolve()
    if not gtf.is_file():
        raise GtfformatError(f"GTF not found: {gtf}")

    tss_counts: dict[tuple[str, int], tuple[int, int]] = {}
    tes_counts: dict[tuple[str, int], tuple[int, int]] = {}

    with gtf.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                continue
            chrom, start_s, end_s, strand = cols[0], cols[3], cols[4], cols[6]
            if strand not in {"+", "-"}:
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            # Match C++ build_tsstes coordinate convention.
            p1 = (chrom, start + 1)
            p2 = (chrom, end)
            if strand == "+":
                pos = tss_counts.get(p1, (0, 0))
                tss_counts[p1] = (pos[0] + 1, pos[1])
                pos = tes_counts.get(p2, (0, 0))
                tes_counts[p2] = (pos[0] + 1, pos[1])
            else:
                neg = tss_counts.get(p2, (0, 0))
                tss_counts[p2] = (neg[0], neg[1] + 1)
                neg = tes_counts.get(p1, (0, 0))
                tes_counts[p1] = (neg[0], neg[1] + 1)

    lines: list[str] = []
    for (chrom, pos), (plus, minus) in sorted(tss_counts.items()):
        lines.append(f"TSS {chrom} {pos} {plus} {minus}")
    for (chrom, pos), (plus, minus) in sorted(tes_counts.items()):
        lines.append(f"TES {chrom} {pos} {plus} {minus}")
    return ("\n".join(lines) + "\n") if lines else ""


def run_get_cov(gtfformat_bin: Path, gtf: Path, out_tsv: Path) -> None:
    """
    Run ``gtfformat get-cov`` via subprocess to write coverage TSV to ``out_tsv``.

    Raises:
        GtfformatError: Missing GTF or non-zero exit from gtfformat.
    """
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


def run_filter_chrom(
    gtfformat_bin: Path | None,
    in_gtf: Path,
    chrom_list_file: Path,
    out_gtf: Path,
) -> None:
    """
    Write ``out_gtf`` keeping only features whose ``transcript_id`` lies on an allowed chromosome.

    Reads nonempty lines from ``chrom_list_file`` into an allowed set. First pass marks transcript_ids
    on allowed chroms; second pass copies only lines for those transcripts (and matching genes).
    Header/comment lines are copied. ``gtfformat_bin`` is ignored (pure Python).
    """
    _ = gtfformat_bin
    in_gtf = in_gtf.resolve()
    chrom_list_file = chrom_list_file.resolve()
    out_gtf = out_gtf.resolve()
    if not in_gtf.is_file():
        raise GtfformatError(f"GTF not found: {in_gtf}")
    if not chrom_list_file.is_file():
        raise GtfformatError(f"Chromosome list not found: {chrom_list_file}")

    allowed: set[str] = set()
    with chrom_list_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            chrom = raw.strip()
            if chrom:
                allowed.add(chrom)

    keep_tx: set[str] = set()
    with in_gtf.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                continue
            chrom, strand, attrs = cols[0], cols[6], cols[8]
            if strand == "." or chrom not in allowed:
                continue
            txid = parse_transcript_id(attrs)
            if txid:
                keep_tx.add(txid)

    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    with in_gtf.open("r", encoding="utf-8", errors="replace") as src, out_gtf.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            line = raw.rstrip("\n")
            if not line:
                dst.write("\n")
                continue
            if line.startswith("#"):
                dst.write(raw)
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            txid = parse_transcript_id(cols[8])
            if txid and txid in keep_tx:
                dst.write(raw)


def run_update_transcript_cov(
    gtfformat_bin: Path | None,
    in_gtf: Path,
    predictions_tsv: Path,
    out_gtf: Path,
) -> None:
    """
    Copy ``in_gtf`` to ``out_gtf``, replacing transcript ``coverage`` / ``cov`` from ``predictions_tsv``.

    TSV format: column 0 = ``transcript_id``, column 1 = float score; accepts optional header row if
    first row’s second column is not float. Non-transcript lines pass through unchanged.
    ``gtfformat_bin`` is ignored (pure Python).
    """
    _ = gtfformat_bin
    in_gtf = in_gtf.resolve()
    predictions_tsv = predictions_tsv.resolve()
    out_gtf = out_gtf.resolve()
    if not in_gtf.is_file():
        raise GtfformatError(f"GTF not found: {in_gtf}")
    if not predictions_tsv.is_file():
        raise GtfformatError(f"Predictions TSV not found: {predictions_tsv}")

    pred_data: dict[str, float] = {}
    with predictions_tsv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            first = next(reader)
        except StopIteration:
            first = []
        rows_iter = []
        if first:
            parsed_first = None
            if len(first) >= 2:
                try:
                    parsed_first = float(first[1])
                except ValueError:
                    parsed_first = None
            # C++ expects a header; accept either with/without header.
            if parsed_first is not None and first[0]:
                pred_data[first[0]] = parsed_first
            rows_iter = list(reader)
        for row in rows_iter:
            if len(row) < 2:
                continue
            txid = row[0].strip()
            if not txid:
                continue
            try:
                pred_data[txid] = float(row[1])
            except ValueError:
                continue

    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    with in_gtf.open("r", encoding="utf-8", errors="replace") as src, out_gtf.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                dst.write(raw)
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                dst.write(raw)
                continue
            if cols[2] == "transcript":
                txid = parse_transcript_id(cols[8])
                if txid in pred_data:
                    cols[8] = _replace_or_append_coverage(cols[8], pred_data[txid])
                    dst.write("\t".join(cols) + "\n")
                    continue
            dst.write(raw)
