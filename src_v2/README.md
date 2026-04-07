# Telos v2 (In Progress)

This directory contains a clean reimplementation path for Telos that is isolated from the legacy `src/` code.

## Goals

- Keep the two-stage Telos logic intact:
  - Stage I: score TSS/TES sites from BAM + GTF-derived candidates.
  - Stage II: score transcripts using Stage I signals + transcript-level features.
- Provide a user-friendly single CLI entry point with minimal required commands.
- Produce clear, primary outputs:
  - ranked transcript table
  - filtered GTF
  - scored TSS/TES table
- Support multi-annotation experiments without ad hoc scripts.

## Status

Scaffold only. No production behavior has been moved from `src/` yet.

## Proposed package layout

```
src_v2/
  telos_v2/
    __init__.py
    cli.py
    config_models.py
    commands/
      __init__.py
      train.py
      predict.py
      benchmark.py
```

See `docs/Telos-v2-implementation-plan.md` for complete step-by-step implementation details.
