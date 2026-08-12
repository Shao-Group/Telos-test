# Agent handoff — paper reproduction repo

## Roles

| Component | Role |
|-----------|------|
| This repo | Paper experiments, benchmark, evaluation, plotting |
| Product **Telos** | `train` / `predict` (pin a checkout; do not upstream paper extras) |

## Status

- Package layout: `src/telos_repro` + product `telos.*` helpers.
- Public path: install → build bundles (Snakemake) → `telos-repro run` → `telos-repro plot` (see README).
- Machine paths: `configs/paths.yaml` (from `paths.example.yaml`) or env vars / `scripts/parity/local.env`.

## Pipeline

1. Configure `configs/paths.yaml` (`bundles_root`, `telos_checkout`, `gffcompare_bin`).
2. Build or point at bundles (`docs/DATA.md`).
3. `telos-repro run cross_annotation_repro -- …`
4. `telos-repro plot cross_annotation -- --root runs/cross_annotation_repro`

## Rules

1. Write outputs under this clone (`runs/`, `plot_v2/`).
2. Train/predict only via product `telos`.
3. Keep absolute lab paths out of committed configs and scripts.
