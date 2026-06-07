# Telos v2 — modules and files

This document is a **high-level map** of the `telos_v2` Python package under `src_v2/telos_v2/`. For **line-by-line behavior**, see the docstrings inside each file (they are written to be granular).

## Package role

Telos v2 runs a **two-stage** transcript boundary and ranking pipeline:

1. **Stage I** scores candidate TSS/TES sites from an assembly GTF + aligned BAM (random forest and XGBoost site classifiers).
2. **Stage II** ranks full transcripts using site probabilities, transcript-level coverage features, and a **LightGBM** model trained with labels from a **gffcompare `.tmap`** (training only).

**Benchmark** mode trains once, predicts on many test bundles, optionally runs transcript-level PR via gffcompare + Python gtfcuff-style metrics, and writes `reports/benchmark_summary.csv`.

---

## Top-level files (`telos_v2/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package version string. |
| `cli.py` | `argparse` entrypoint: subcommands `train`, `predict`, `benchmark`, `benchmark-matrix`; builds IO dataclasses and dispatches to `commands/*`. |
| `config_loader.py` | Resolve default Stage I YAML path; load YAML/JSON config mappings; nested dict access helper. |
| `config_models.py` | Frozen dataclasses `TrainIO`, `PredictIO`, `BenchmarkIO` (and shared `RunIO`) describing CLI/run inputs. |
| `config_validation.py` | Strict validation of Stage I config shape and benchmark YAML shape (allowed keys, types). |
| `pipeline_core.py` | Shared Stage I preparation: build `Stage1RuntimeConfig` from YAML + CLI overrides; load candidates + cov table; run BAM feature extraction into a dataframe. |
| `gtf_attributes.py` | Parse `transcript_id` from GTF attribute column 9 (quoted form). |

---

## `telos_v2/commands/`

Thin wrappers so `cli.py` stays small. Each handler validates config, runs preflight, calls `benchmark` or `pipeline_core` + `models/*`.

| File | Purpose |
|------|---------|
| `train.py` | End-to-end training: Stage I per site-type and backend, write `sites.scored.tsv`, Stage II per backend, write ranked transcript TSVs. |
| `predict.py` | Load saved models; Stage I scoring + Stage II ranking for a new BAM/GTF. |
| `benchmark.py` | Re-exports `run_benchmark` from `benchmark.orchestrator` for CLI symmetry. |
| `benchmark_matrix.py` | Re-exports `run_benchmark_matrix` from `benchmark.matrix`. |
| `__init__.py` | Package marker for command handlers. |

---

## `telos_v2/benchmark/`

Orchestration and reporting for multi-dataset benchmarks.

| File | Purpose |
|------|---------|
| `orchestrator.py` | `run_benchmark`: parse YAML, optional train or `model_dir` skip, loop tests with `run_predict`, optional Stage I test AUPR, sklearn-on-tmap or gffcompare transcript PR, write summary CSV. |
| `matrix.py` | Fixed **benchmark matrix** over bundle layout: resolve `data/bundles`, build benchmark YAML from data type + annotation refs, write `generated_benchmark.yaml`, call `run_benchmark`. |
| `report.py` | CSV summary schema helpers: stub rows for bad configs, `write_benchmark_summary_csv` with stable column order. |
| `util.py` | Path coercion for benchmark dicts; rounding float metrics in result rows. |
| `stage1_metrics.py` | After predict, compute **test-split** Stage I AUPR by joining scored sites to assembly coverage and reference sites. |
| `transcript_pr_runner.py` | Adapter from benchmark row context into `evaluation.transcript_pr_pipeline.run_transcript_pr_benchmark`; merges PR metrics back into the row. |
| `__init__.py` | Lazy exports: `run_benchmark`, `run_benchmark_matrix` (avoids loading `pysam`/train stack when importing e.g. `benchmark.matrix` alone). |

---

## `telos_v2/validation/`

| File | Purpose |
|------|---------|
| `preflight.py` | Fail-fast checks: BAM+index, coordinate sort, GTF shape, training tmap presence, predict model artifact presence, benchmark config non-empty. Defines `RunLayout` directory convention. |

---

## `telos_v2/evaluation/`

| File | Purpose |
|------|---------|
| `transcript_pr_pipeline.py` | Full transcript PR: inject scores into GTF, optional chromosome filter, run gffcompare, Python PR/AUC from tmap-like inputs, optional plots and tables. |
| `pr_ranking.py` | Generic precision–recall math and plotting vs a baseline abundance column (used where sklearn-style PR is enough). |
| `benchmark_sklearn_metrics.py` | Fast diagnostic: merge ranked TSV with a static bundle tmap and compute sklearn metrics (not the gffcompare PR path). |
| `__init__.py` | Re-exports selected symbols from `pr_ranking`. |

---

## `telos_v2/backends/`

Pure-Python stand-ins for external RNA-seq helper tools where applicable.

| File | Purpose |
|------|---------|
| `gtfformat.py` | GTF parsing, coverage extraction, `filter-chrom`, `update-transcript-cov` behaviors used by training and PR. |
| `gtfcuff.py` | ROC/AUC-style computation from classification tables (Python port of the old binary workflow’s numerics). |

---

## `telos_v2/candidates/`

| File | Purpose |
|------|---------|
| `extract.py` | Derive candidate TSS/TES sites from transcript boundaries in the assembly GTF. |
| `load.py` | `load_candidates` and `load_transcript_cov_dataframe` as the canonical entrypoints for Stage I/II tables. |
| `tsstes.py` | Low-level TSSTES-style site extraction helpers. |

---

## `telos_v2/labels/`

| File | Purpose |
|------|---------|
| `site_labels.py` | Reference sites from ref GTF; proximity labeling for Stage I; chromosome name normalization. |
| `transcript_labels.py` | Transcript-level label helpers where needed for Stage II. |

---

## `telos_v2/features/`

| File | Purpose |
|------|---------|
| `stage1.py` | BAM-derived features around each candidate site (windows, coverage, splice signals, etc.). |
| `stage2.py` | Features for the transcript ranking frame. |

---

## `telos_v2/models/`

Training and inference for Stage I and II.

| File | Purpose |
|------|---------|
| `__init__.py` | Constants (backend tags, filenames for joblib/json outputs, column names). |
| `chrom_split.py` | Parse `split_policy` strings into numeric chromosome ranges for train/holdout. |
| `stage1_train.py` | Train RF/XGB per site type; save bundles. |
| `stage1_predict.py` | Score all candidate rows with saved Stage I models; write TSV. |
| `stage2_train.py` | Build Stage II training frame from cov + sites + tmap; train LightGBM; write model + ranked train predictions. |
| `stage2_predict.py` | Load Stage II model; score inference frame; write ranked TSV. |

---

## `telos_v2/postprocess/`

| File | Purpose |
|------|---------|
| `filter_gtf.py` | Optional GTF filtering utilities (not always on the hot path for CLI). |

---

## `telos_v2/reporting/`

Placeholder package for future reporting helpers; core runs stay artifact-focused.

---

## Related paths outside this package tree

- `src_v2/configs/stage1.defaults.yaml` — default Stage I/feature/training knobs referenced by `config_loader.default_stage1_config_path()`.
- `docs/benchmark-matrix-convention.md` — human-readable contract for bundle layout + benchmark-matrix CLI (if present in repo).

Implementation details for each function live in **granular docstrings** in the source files; this document only maps modules and files.
