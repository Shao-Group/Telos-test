from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re


@dataclass(frozen=True)
class CandidateSite:
    transcript_id: str
    site_type: str  # TSS or TES
    chrom: str
    position: int
    strand: str


_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')


def _parse_transcript_id(attr: str) -> str | None:
    m = _TX_ID_RE.search(attr)
    return m.group(1) if m else None


def extract_candidate_sites_from_gtf(gtf_path: Path) -> list[CandidateSite]:
    """
    Build transcript-end candidates directly from transcript records in GTF.
    For each transcript, emit:
      - TSS = start on '+' and end on '-'
      - TES = end on '+' and start on '-'
    """
    candidates: list[CandidateSite] = []
    with gtf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = cols
            if feature != "transcript":
                continue
            if strand not in {"+", "-"}:
                continue
            tx_id = _parse_transcript_id(attrs)
            if not tx_id:
                continue

            s = int(start)
            e = int(end)
            if strand == "+":
                tss_pos = s
                tes_pos = e
            else:
                tss_pos = e
                tes_pos = s

            candidates.append(
                CandidateSite(
                    transcript_id=tx_id,
                    site_type="TSS",
                    chrom=chrom,
                    position=tss_pos,
                    strand=strand,
                )
            )
            candidates.append(
                CandidateSite(
                    transcript_id=tx_id,
                    site_type="TES",
                    chrom=chrom,
                    position=tes_pos,
                    strand=strand,
                )
            )
    return candidates


def write_candidates_tsv(candidates: list[CandidateSite], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["transcript_id", "site_type", "chrom", "position", "strand"])
        for c in candidates:
            writer.writerow([c.transcript_id, c.site_type, c.chrom, c.position, c.strand])
