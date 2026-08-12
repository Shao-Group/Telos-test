# Goldens / reference metrics

Public reproduction regenerates metrics under `runs/` and figures under `plot_v2/`.
Optional **reference summaries** (for maintainer drift checks) can be pointed to via
`goldens_runs_root` / `goldens_figures_root` in `configs/paths.yaml`.

Small committed snapshots (when present) live under `goldens/` in this repo.

## Provenance (refactor freeze)

| Item | Value |
|------|-------|
| Frozen source SHA (pre-refactor) | `ed3c8b5d6f77ef99d5deeb5e08ee7f4ea2989439` |
| Tag | `pre-repro-refactor-20260806` |
| Product Telos pin (at freeze) | `a9d37b752f79f3905e95d0203c3c04ed0cd73e55` |
| Stage I config | `configs/stage1.defaults.yaml` |
| Stage I config SHA256 | `fa93d3ec2417abc7f8124cd5a6c335814870085fb452f7e3ce7b94e0628bfbc0` |
| gffcompare | any `v0.11.2` on `PATH` (or set `gffcompare_bin`) |

## Typical experiment run dirs (under a reference `runs/` tree)

| Experiment dir | Notes |
|----------------|-------|
| `cross_annotation_repro` | Main paper grid |
| `novel_phase_a_cross_annotation` | Novel Phase A |
| `human_gencode_feature_window` | Window sweep |
| `mouse_cross_species_gencode` | Mouse |
| `tissue_human_gencode` | Tissue |
| `stage1_feature_importance_gencode` | Feature importance |
| `refseq_novel_eval` | RefSeq-novel |

Compare `reports/benchmark_summary.csv` files; PDF byte identity is not required.

## Local committed smoke CSVs

| Path | Role |
|------|------|
| `goldens/local_tier1_retrain/` | Historical local retrain notes (not the public gate) |

## Next

1. Public data path: [`docs/DATA.md`](../docs/DATA.md)
2. Figures: [`figures/CATALOG.md`](../figures/CATALOG.md)
