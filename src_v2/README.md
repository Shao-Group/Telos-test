# Telos v2

Python implementation of the two-stage Telos pipeline (Stage I: TSS/TES sites; Stage II: transcripts), isolated from legacy `src/`. External **rnaseqtools** binaries are not required for core train/predict; transcript-level benchmark PR uses **gffcompare** on PATH (or `GFFCOMPARE` / YAML).

## Layout

```
src_v2/
  configs/
    stage1.defaults.yaml    # default train/feature/LightGBM thread settings
  telos_v2/
    cli.py                  # telos_v2 train | predict | benchmark | benchmark-matrix
    pipeline_core.py        # shared Stage I inputs (candidates + cov + features)
    gtf_attributes.py       # shared GTF attribute parsing (e.g. transcript_id)
    benchmark/              # benchmark orchestration (train once, predict many, summary CSV)
    commands/               # thin CLI entrypoints
    evaluation/             # transcript PR (gffcompare path) + sklearn PR helpers
    backends/               # pure-Python gtfformat/gtfcuff-style helpers
    ...
```

## CLI

Run with `PYTHONPATH=src_v2` (or install the package):

```bash
cd src_v2
conda run -n irtesam-berth python -m telos_v2.cli train --bam ... --gtf ... --ref-gtf ... --tmap ...
conda run -n irtesam-berth python -m telos_v2.cli predict --bam ... --gtf ... --model-dir ...
conda run -n irtesam-berth python -m telos_v2.cli benchmark --config path/to/benchmark.yaml [--outdir DIR]
```

- **train**: `--config` defaults to `configs/stage1.defaults.yaml` beside this package if omitted.
- **benchmark**: if `train.config` is omitted in YAML, the same default Stage I config is used for training/predict feature extraction.

## Stable I/O contract

| Mode | Inputs | Outputs |
|------|--------|---------|
| **train** | `bam`, assembly `gtf`, `ref_gtf`, `tmap`, optional `--config`, optional `--outdir` | `models/*`, `predictions/sites.scored.tsv`, `predictions/transcripts.ranked.{rf,xgb}.tsv` |
| **predict** | `bam`, assembly `gtf`, `model_dir`, optional `--config`, optional `--outdir` | `predictions/sites.scored.tsv`, `predictions/transcripts.ranked.{rf,xgb}.tsv` |
| **benchmark** | YAML: `train`, `tests[]`, optional `analysis`, `execution` | **`reports/benchmark_summary.csv`** (required); transcript PR extras depend on `analysis.benchmark_mode` |

### Benchmark analysis

- **`analysis.benchmark_mode`**: `minimal` (default if omitted in code paths that default) | `full`
  - **minimal**: transcript PR metrics computed using an **ephemeral** temp workdir (no persistent `reports/pr/work_*` tree unless `analysis.debug.keep_pr_work: true`).
  - **full**: persistent `reports/pr/` workdirs; honors `pr_vs_baseline.save_pr_tables` and `plot`.
- **`analysis.debug.keep_pr_work`**: if `true`, keep on-disk PR work under `reports/pr/` even when mode is `minimal`.
- **gffcompare**: must be on **`PATH`**, or set env **`GFFCOMPARE`**, or set `analysis.pr_vs_baseline.gffcompare_bin`. There is **no** machine-specific fallback path in code.

### Config (`stage1.defaults.yaml`)

- **`stage1.training.lightgbm.n_jobs`**: LightGBM thread count for Stage II (default `-1`; use a small integer on shared clusters).
- Strict YAML validation: unknown keys under `stage1` / benchmark sections raise errors.

## Portable tool resolution

- **gffcompare**: PATH / `GFFCOMPARE` / YAML only.
- **gtfformat** (optional legacy subprocess): `TELOS_GTFFORMAT`, `PATH`, or repo-relative `tools/rnaseqtools/gtfformat/gtfformat` if present.

## Tests

```bash
# unittest (no extra deps)
PYTHONPATH=src_v2 python -m unittest discover -s tests/test_telos_v2 -p 'test_*.py' -v

# pytest (optional: pip install -r requirements-dev.txt)
PYTHONPATH=src_v2 pytest tests/test_telos_v2 -q
```

## Further docs

- Repo root `docs/Telos-v2-implementation-plan.md` — historical handoff notes.
