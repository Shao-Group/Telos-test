"""
Parse TSSTES-style text lines into :class:`~telos_v2.candidates.extract.CandidateSite` rows.

Used when consuming output that looks like ``gtfformat TSSTES`` (site type, chrom, pos, strand counts).
"""

from __future__ import annotations

from telos_v2.candidates.extract import CandidateSite

# TSSTES stream lines do not carry transcript_id; use a constant placeholder in CandidateSite rows.
_TSSTES_TRANSCRIPT_PLACEHOLDER = "."


def parse_tsstes_output(text: str) -> list[CandidateSite]:
    """
    Parse whitespace-separated TSSTES lines from ``text`` (e.g. subprocess stdout).

    Expected columns per line: ``<TSS|TES> <chrom> <pos> <plus_strand_count> <minus_strand_count>``.

    Strand selection:

    - If plus-count > minus-count → strand ``+``.
    - If plus-count < minus-count → strand ``-``.
    - If tied → emit **two** :class:`CandidateSite` rows at the same locus, one ``+`` and one ``-``.

    Invalid lines (too few tokens, bad ints, wrong site type) are skipped. Returns TSS sites followed
    by TES sites (each group in discovery order, not sorted by coordinate).
    """
    tss_sites: list[CandidateSite] = []
    tes_sites: list[CandidateSite] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        site_type, chrom, pos_s, pos_cnt_s, neg_cnt_s = parts[0], parts[1], parts[2], parts[3], parts[4]
        if site_type not in ("TSS", "TES"):
            continue
        try:
            pos = int(pos_s)
            pos_strand_count = int(pos_cnt_s)
            neg_strand_count = int(neg_cnt_s)
        except ValueError:
            continue
        if pos_strand_count > neg_strand_count:
            strand = "+"
        elif pos_strand_count < neg_strand_count:
            strand = "-"
        else:
            for s in ("+", "-"):
                site = CandidateSite(
                    transcript_id=_TSSTES_TRANSCRIPT_PLACEHOLDER,
                    site_type=site_type,
                    chrom=chrom,
                    position=pos,
                    strand=s,
                )
                if site_type == "TSS":
                    tss_sites.append(site)
                else:
                    tes_sites.append(site)
            continue
        site = CandidateSite(
            transcript_id=_TSSTES_TRANSCRIPT_PLACEHOLDER,
            site_type=site_type,
            chrom=chrom,
            position=pos,
            strand=strand,
        )
        if site_type == "TSS":
            tss_sites.append(site)
        else:
            tes_sites.append(site)
    return tss_sites + tes_sites
