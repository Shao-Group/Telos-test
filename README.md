# Telos-repro

Paper **reproduction add-on** for product Telos. Telos trains and predicts; this repo adds
experiments, benchmarking, plotting, golden parity, and FASTQ→bundle rebuild.


| Role                     | Path                                                           |
| ------------------------ | -------------------------------------------------------------- |
| This repo                | `/datadisk1/ixk5174/project_repo/Telos-repro`                  |
| Product Telos (pinned)   | `/datadisk1/ixk5174/project_repo/Telos`                        |
| Frozen goldens / bundles | `/datadisk1/ixk5174/project_repo/Telos-test` (**do not edit**) |


See `[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)`, `[docs/DATA.md](docs/DATA.md)`,
`[docs/HANDOFF.md](docs/HANDOFF.md)`.

## Setup

```bash
conda activate irtesam-berth
pip install -e .
cp configs/paths.example.yaml configs/paths.yaml
```

```bash
PYTHONPATH=src:/path/to/Telos/src python -m telos_repro list
```



## Commands

```bash
telos train …
telos predict …

telos-repro list
telos-repro status
telos-repro run cross_annotation_repro -n
telos-repro plot cross_annotation
telos-repro parity 1 --local-summary runs/…/benchmark_summary.csv
```



## Layout

```text
src/telos_repro/   CLI + paper benchmark/eval/plot + backend → telos
src/experiments/   experiment drivers
configs/           registry, paths, stage1, parity
workflow/          FASTQ → bundles
goldens/ figures/ docs/
scripts/parity/ scripts/data/ scripts/legacy/
```



## Public data

