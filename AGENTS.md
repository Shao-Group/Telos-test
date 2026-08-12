# AGENTS.md — Telos paper reproduction

1. Read [`docs/HANDOFF.md`](docs/HANDOFF.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
2. Follow [`.cursor/plans/paper_repro_repo_design_390b2976.plan.md`](.cursor/plans/paper_repro_repo_design_390b2976.plan.md) when present.
3. Train/predict → product `telos` only (`telos_repro.backend`). Shared helpers import from `telos.*`.
4. Paper-only code lives in `src/telos_repro/` (benchmark, evaluation, plotting, gtfcuff, novel labels).
5. Do **not** merge paper extras upstream into the product Telos repo unless asked.
6. Machine-local paths belong in gitignored `configs/paths.yaml` or `scripts/parity/local.env` — never hard-code lab absolute paths in committed files.
7. Public data path: [`docs/DATA.md`](docs/DATA.md).
