# Parity / smoke scripts

These helpers are optional. The public path is `telos-repro run` / `plot` (see README).

| Script | Purpose |
|--------|---------|
| `_env.sh` | Shared path bootstrap (`REPO_ROOT`, `TELOS_SRC`, `TELOS_BUNDLES_ROOT`, `GFFCOMPARE`) |
| `run_parity_tier1_tmux.sh` | Small benchmark smoke cell |
| `run_parity_multi_dataset_tmux.sh` | Multi-sample SR smoke (needs `REF_RUNS_ROOT` for Phase A) |
| `run_cross_annotation_gencode_gencode_nocache.sh` | gencode→gencode grid, no Stage I cache |
| `qc_stage1_features.py` | Fresh Stage I feature anomaly checks |
| `compare_to_telos_test_runs.py` | Diff local vs reference `benchmark_summary.csv` trees |
| `diag_*.py` | PR replay diagnostics |

Configure via `configs/paths.yaml`, environment variables, or copy:

```bash
cp scripts/parity/local.env.example scripts/parity/local.env
# edit local.env (gitignored)
```

Historical `telos_v2` scripts: `scripts/legacy/parity_historical/`.
