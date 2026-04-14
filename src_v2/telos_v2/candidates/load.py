from __future__ import annotations

from pathlib import Path
from typing import Any

from telos_v2.backends.gtfformat import resolve_gtfformat_binary, run_get_cov, run_tsstes
from telos_v2.candidates.extract import CandidateSite, extract_candidate_sites_from_gtf
from telos_v2.candidates.tsstes import parse_tsstes_output
from telos_v2.config_loader import get_nested


def _norm_source(raw: Any) -> str:
    s = str(raw or "gtfformat").strip().lower()
    if s in ("", "default"):
        return "gtfformat"
    return s


def resolve_gtfformat_for_config(cfg_map: dict[str, Any]) -> Path:
    explicit = get_nested(cfg_map, ["rnaseqtools", "gtfformat_bin"], None)
    if explicit is not None and str(explicit).strip() == "":
        explicit = None
    return resolve_gtfformat_binary(Path(explicit) if explicit else None)


def load_candidates(cfg_map: dict[str, Any], gtf_path: Path) -> list[CandidateSite]:
    source = _norm_source(get_nested(cfg_map, ["candidates", "source"], "gtfformat"))
    if source in ("gtf_transcript", "transcript", "gtf"):
        return extract_candidate_sites_from_gtf(gtf_path)
    if source in ("gtfformat", "tsstes"):
        gbin = resolve_gtfformat_for_config(cfg_map)
        text = run_tsstes(gbin, gtf_path)
        return parse_tsstes_output(text)
    raise ValueError(
        f"Unknown candidates.source={source!r}; use 'gtfformat' or 'gtf_transcript'."
    )


def gtfformat_needed_for_run(cfg_map: dict[str, Any]) -> bool:
    source = _norm_source(get_nested(cfg_map, ["candidates", "source"], "gtfformat"))
    run_cov = bool(get_nested(cfg_map, ["rnaseqtools", "run_get_cov"], True))
    if run_cov:
        return True
    return source in ("gtfformat", "tsstes")


def run_transcript_cov_table(cfg_map: dict[str, Any], gtf_path: Path, out_tsv: Path) -> None:
    """Write gtfformat get-cov transcript table when rnaseqtools.run_get_cov is true."""
    if not bool(get_nested(cfg_map, ["rnaseqtools", "run_get_cov"], True)):
        return
    gbin = resolve_gtfformat_for_config(cfg_map)
    run_get_cov(gbin, gtf_path, out_tsv)
