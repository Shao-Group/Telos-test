from __future__ import annotations

from pathlib import Path

from telos_v2.backends.gtfformat import (
    GtfformatError,
    resolve_gtfformat_binary,
    run_remove_fp,
    run_remove_fp_threshold,
)
from telos_v2.config_loader import get_nested, load_mapping_config
from telos_v2.config_models import FilterGtfIO
from telos_v2.validation.preflight import PreflightError, validate_input_file, validate_input_gtf


def run_filter_gtf(cfg: FilterGtfIO) -> int:
    cfg_map = load_mapping_config(cfg.config_file)
    explicit = cfg.gtfformat_bin
    if explicit is None or (isinstance(explicit, str) and not str(explicit).strip()):
        nested = get_nested(cfg_map, ["rnaseqtools", "gtfformat_bin"], None)
        explicit = nested if nested else None
    if explicit is not None and str(explicit).strip() == "":
        explicit = None
    try:
        gbin = resolve_gtfformat_binary(Path(explicit) if explicit else None)
    except GtfformatError as exc:
        print(f"[telos_v2] filter-gtf: {exc}")
        return 2

    try:
        validate_input_gtf(cfg.gtf)
        validate_input_file(cfg.predictions, "Predictions TSV")
    except PreflightError as exc:
        print(f"[telos_v2] filter-gtf preflight failed: {exc}")
        return 2

    mode = str(cfg.mode).strip().lower()
    try:
        if mode == "exact":
            run_remove_fp(gbin, cfg.gtf, cfg.predictions, cfg.out_gtf, cfg.hard_mode)
        elif mode == "threshold":
            run_remove_fp_threshold(
                gbin,
                cfg.gtf,
                cfg.predictions,
                cfg.out_gtf,
                cfg.hard_mode,
                int(cfg.bp_threshold),
            )
        else:
            print(f"[telos_v2] filter-gtf: unknown mode {cfg.mode!r}; use 'exact' or 'threshold'.")
            return 2
    except GtfformatError as exc:
        print(f"[telos_v2] filter-gtf failed: {exc}")
        return 2

    print("[telos_v2] filter-gtf complete")
    print(f"  in_gtf={cfg.gtf}")
    print(f"  predictions={cfg.predictions}")
    print(f"  out_gtf={cfg.out_gtf}")
    print(f"  mode={mode} hard_mode={cfg.hard_mode}" + (f" bp_threshold={cfg.bp_threshold}" if mode == "threshold" else ""))
    return 0
