from __future__ import annotations

from telos_v2.candidates.extract import CandidateSite

# TSSTES lines have no per-transcript id; match legacy extract_features.load_candidate_sites.
_TSSTES_TRANSCRIPT_PLACEHOLDER = "."


def parse_tsstes_output(text: str) -> list[CandidateSite]:
    """
    Parse stdout of `gtfformat TSSTES <gtf>`.

    Each line: TSS|TES <chrom> <pos> <pos_strand_count> <neg_strand_count>
    Strand is inferred like legacy code: tie -> emit both + and -.
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
