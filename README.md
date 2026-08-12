# Telos paper reproduction

Reproduce the **Telos paper benchmarks and figures** on your machine.

This repo orchestrates the experiment grid and plotting. Training and prediction
come from the product [Telos](https://github.com/) package.

```text
public FASTQ  →  bundles (BAM + assemblies + tmap)
              →  telos-repro run <experiment>
              →  runs/*/reports/benchmark_summary.csv
              →  telos-repro plot <figure>  →  PDF
```

---

## Requirements

- Python ≥ 3.10 and a conda/venv with Telos dependencies (pysam, scikit-learn,
  xgboost, lightgbm, pandas, PyYAML, …)
- A checkout of product **Telos** (sibling directory is fine)
- [gffcompare](https://github.com/gpertea/gffcompare) on `PATH` (or set its path below)
- [Snakemake](https://snakemake.readthedocs.io/) + conda (to build bundles from FASTQ)
- Substantial disk and CPU for the full grid; start with a small slice (below)

---

## Quick start

### 1. Install

```bash
git clone <this-repo-url> Telos-repro
cd Telos-repro

# Product Telos next to this repo (or anywhere you prefer)
#   ../Telos

conda activate <your-env>
pip install -e .
pip install -e ../Telos
# Alternative to installing Telos editable:
#   export PYTHONPATH="$PWD/src:../Telos/src:$PYTHONPATH"

cp configs/paths.example.yaml configs/paths.yaml
```

Edit `configs/paths.yaml` for your machine (this file is local; do not commit it):

```yaml
telos_checkout: /absolute/path/to/Telos
bundles_root:   /absolute/path/to/data/bundles   # after Snakemake
runs_root:      /absolute/path/to/Telos-repro/runs
figures_root:   /absolute/path/to/Telos-repro/plot_v2
gffcompare_bin: /absolute/path/to/gffcompare
backend: telos
```

Confirm the CLI:

```bash
telos-repro list
```

### 2. Build input bundles

Paper runs need per-sample **bundles**: aligned BAM, assembler GTFs, gffcompare
`.tmap` files, and `bundle_manifest.yaml`.

1. Download reads by accession — lists and layout in [`docs/DATA.md`](docs/DATA.md).
2. Configure and run the Snakemake workflow:

```bash
cp workflow/config/config.example.yaml workflow/config/config.yaml
cp workflow/config/samples.example.yaml workflow/config/samples.yaml
# edit samples + genome/annotation paths

snakemake -s workflow/Snakefile \
  --configfile workflow/config/config.yaml \
  --use-conda --cores 16
```

3. Point `bundles_root` (or `export TELOS_BUNDLES_ROOT=…`) at the resulting
   `data/bundles` tree.

Details: [`docs/DATA.md`](docs/DATA.md), [`workflow/README.md`](workflow/README.md).

### 3. Run an experiment

Experiments write under `runs/<experiment_id>/`. The CSV summaries are what the
figures read.

**Main paper benchmark** (cross-annotation). Start with same-annotation
gencode→gencode:

```bash
# Dry-run
telos-repro run cross_annotation_repro -n -- \
  --annotation-pairs gencode-gencode

# Full run for that pair
telos-repro run cross_annotation_repro -- \
  --outdir runs/cross_annotation_repro \
  --annotation-pairs gencode-gencode \
  --stage1-config configs/stage1.defaults.yaml \
  --max-parallel-trains 4 \
  --max-parallel-cells 4 \
  --max-parallel-tests 4 \
  --total-cpus 64
```

Useful slices:

| Goal | Extra args |
|------|------------|
| One annotation pair | `--annotation-pairs gencode-gencode` |
| Same-annotation diagonal only | `--only-same-annotation` |
| Full cross-annotation grid | omit `--annotation-pairs` (large) |

Primary outputs:

```text
runs/cross_annotation_repro/
  _cross_annotation_shared_train/<modality>__train_<anno>/models/
  <modality>__train_<anno>__test_<anno>/reports/benchmark_summary.csv
```

**Other paper experiments** (same pattern):

```bash
telos-repro list -v          # ids, default outdirs, linked plots
telos-repro status           # which summaries already exist locally

telos-repro run novel_phase_a_cross_annotation -- …
telos-repro run mouse_cross_species_gencode -- …
telos-repro run tissue_human_gencode -- …
telos-repro run human_gencode_feature_window -- …
telos-repro run stage1_feature_importance_gencode -- …
telos-repro run evaluate_refseq_novel -- …
```

### 4. Make figures

After the matching `runs/` summaries exist:

```bash
telos-repro plot cross_annotation -- --root runs/cross_annotation_repro

# Or regenerate every registered figure that has inputs
telos-repro plot all
```

| Plot id | Typical `--root` / run dir |
|---------|----------------------------|
| `cross_annotation` | `runs/cross_annotation_repro` |
| `novel_phase_a_cross` | `runs/novel_phase_a_cross_annotation` |
| `mouse` | `runs/mouse_cross_species_gencode` |
| `tissue` | `runs/tissue_human_gencode` |
| `window` | `runs/human_gencode_feature_window` |
| `feature_importance` | `runs/stage1_feature_importance_gencode` |
| `refseq_novel` | `runs/refseq_novel_eval` |

PDFs land under `plot_v2/` (or `figures_root` in `paths.yaml`). Command map:
[`figures/CATALOG.md`](figures/CATALOG.md).

Numbers live in `benchmark_summary.csv`. Regenerated PDFs need not match a prior
run byte-for-byte.

---

## Experiment → figure map

| Experiment (`telos-repro run …`) | Figure (`telos-repro plot …`) |
|----------------------------------|-------------------------------|
| `cross_annotation_repro` | `cross_annotation` |
| `novel_phase_a_cross_annotation` | `novel_phase_a_cross` |
| `mouse_cross_species_gencode` | `mouse` |
| `tissue_human_gencode` | `tissue` |
| `human_gencode_feature_window` | `window` |
| `stage1_feature_importance_gencode` | `feature_importance` |
| `evaluate_refseq_novel` | `refseq_novel` |

Registry source: [`configs/experiments.yaml`](configs/experiments.yaml).

---

## How this relates to Telos

| Layer | Role |
|-------|------|
| **Telos** (product) | `train` / `predict`, Stage I–II features and models |
| **This repo** | Bundle matrix, shared-train grid, PR evaluation, CLI, plots |

Orchestration calls Telos through `telos_repro.backend`. You do not need a second
training stack. Package layout: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```text
src/telos_repro/     CLI, Telos backend adapter, benchmark, evaluation, plotting
src/experiments/     Paper experiment entrypoints
configs/             paths template, stage1 defaults, experiment registry
workflow/            FASTQ → bundles (Snakemake)
figures/             Figure catalog
docs/                Data accessions, architecture
```

---

## Further reading

- [`docs/DATA.md`](docs/DATA.md) — accessions, genomes, bundle layout  
- [`workflow/README.md`](workflow/README.md) — Snakemake details  
- [`figures/CATALOG.md`](figures/CATALOG.md) — plot commands and output paths  
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — what lives in this repo vs product Telos  
