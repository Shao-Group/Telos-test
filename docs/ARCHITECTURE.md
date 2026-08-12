# Architecture — Telos + paper reproduction add-on

## Mental model

**Product Telos** trains and predicts; **this repo** adds paper experiments, benchmarking,
plotting, parity checks, and FASTQ→bundle rebuild — without a second train/predict stack.

```text
┌─────────────────────────────────────────────────────────┐
│  telos (product package — pinned checkout)              │
│    telos prepare-gtf | train | predict                  │
│    shared helpers: labels, gtfformat, candidates, …     │
└──────────────────────────▲──────────────────────────────┘
                           │ facade + imports
┌──────────────────────────┴──────────────────────────────┐
│  telos-repro (this repo)                                │
│    telos-repro list | status | run | plot | parity      │
│    paper-only: benchmark, evaluation, plotting, gtfcuff │
│    workflow/  (FASTQ → bundles)                         │
└─────────────────────────────────────────────────────────┘
```

## What users run

| Goal | Command |
|------|---------|
| Train / predict | `telos train` / `telos predict` |
| Paper experiment | `telos-repro run <experiment>` |
| Paper figures | `telos-repro plot <figure>` |
| Golden parity | `telos-repro parity 1 --local-summary …` |
| Low-level benchmark YAML | `python -m telos_repro.pipeline_cli benchmark …` |
| Rebuild bundles | `snakemake -s workflow/Snakefile …` (`docs/DATA.md`) |

## Package layout

```text
src/
  telos_repro/
    backend/           → product telos train/predict
    cli.py             telos-repro entry
    pipeline_cli.py    optional low-level train/predict/benchmark
    benchmark/ evaluation/ plotting/ analysis/
    backends/gtfcuff.py          # paper-only
    labels/novel_vs_refseq.py    # paper-only
    config_loader.py             # telos helpers + repo stage1 path
  experiments/         telos-repro run drivers
configs/               experiments.yaml, paths, stage1, parity
workflow/              bundle rebuild
goldens/ figures/ docs/
scripts/parity/ scripts/data/ scripts/legacy/
```

Shared helpers are imported from **product `telos`** (not forked copies):
`telos.labels`, `telos.backends.gtfformat`, `telos.candidates`, `telos.models`,
`telos.validation`, `telos.config_models`, …

**Not done / not planned:** merging paper extras upstream into the Telos product git repo.

## Migration status

1. ~~Delete vendored train/predict; facade → Telos~~  
2. ~~Single `src/` tree; remove `src_v2/`~~  
3. ~~Rename paper modules into `telos_repro`; use `telos.*` shared helpers~~  
4. Optional later: HPC Tier-2 scaffolding only (do **not** upstream paper extras into product Telos)
