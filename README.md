# Telos — paper reproduction

This repository reproduces the **Telos paper experiments and figures**.

Training and prediction use the product **Telos** package. This repo adds the
benchmark grid, evaluation, and plotting so you can regenerate results end to end.

```text
FASTQ / public accessions  →  bundles (BAM, assembly GTF, tmap)
                           →  telos train / predict  (via this repo’s runners)
                           →  runs/*/reports/benchmark_summary.csv
                           →  telos-repro plot …  →  PDF figures
```

---

## What you need


| Requirement                                         | Notes                                                                                     |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Python ≥ 3.10                                       | Conda env with Telos deps (pysam, scikit-learn, xgboost, lightgbm, pandas, PyYAML, …)     |
| [Telos](https://github.com/) product checkout       | Pinned install or `PYTHONPATH=…/Telos/src`                                                |
| [gffcompare](https://github.com/gpertea/gffcompare) | For transcript PR metrics                                                                 |
| Compute                                             | Full cross-annotation grid is large (many BAM Stage I extracts); start with a small slice |


Optional: an existing bundle tree if you already have aligned assemblies (skips FASTQ rebuild).

---

## 1. Install

```bash
git clone <this-repo-url> Telos-repro
cd Telos-repro

# Product Telos (sibling checkout is fine)
#   ../Telos

conda activate <your-env>
pip install -e .                    # provides `telos-repro`
pip install -e ../Telos             # or: export PYTHONPATH="$PWD/src:../Telos/src:$PYTHONPATH"

cp configs/paths.example.yaml configs/paths.yaml
```

Edit `configs/paths.yaml` (local only; not committed):

```yaml
telos_checkout: /absolute/path/to/Telos
bundles_root:   /absolute/path/to/data/bundles   # after Snakemake, or your existing bundles
runs_root:      /absolute/path/to/Telos-repro/runs
figures_root:   /absolute/path/to/Telos-repro/plot_v2
gffcompare_bin: /absolute/path/to/gffcompare
backend: telos
```

Check the registry:

```bash
telos-repro list
```

---

## 2. Prepare data (bundles)

**Public path (recommended for outsiders):** download reads by accession, then build bundles.

1. Follow accession lists and layout in `[docs/DATA.md](docs/DATA.md)`.
2. Configure and run Snakemake:

```bash
cp workflow/config/config.example.yaml workflow/config/config.yaml
cp workflow/config/samples.example.yaml workflow/config/samples.yaml
# edit samples + genome/annotation paths

snakemake -s workflow/Snakefile \
  --configfile workflow/config/config.yaml \
  --use-conda --cores 16
```

1. Point `bundles_root` (or `TELOS_BUNDLES_ROOT`) at the resulting
  `data/bundles` tree.

Each sample directory should contain a BAM, assembler GTFs, gffcompare `.tmap`
files, and `bundle_manifest.yaml` (see `docs/DATA.md`).

---

## 3. Run experiments (produce numeric results)

Experiments write under `runs/<experiment_id>/` (CSV summaries drive the figures).

### Cross-annotation grid (main paper benchmark)

Dry-run one train→test pair:

```bash
telos-repro run cross_annotation_repro -n -- \
  --annotation-pairs gencode-gencode
```

Run it (retrains shared models per modality, then scores held-out tests):

```bash
# Optional: disable Stage I feature disk cache for a fully fresh extract
unset TELOS_STAGE1_CACHE_DIR

telos-repro run cross_annotation_repro -- \
  --outdir runs/cross_annotation_repro \
  --annotation-pairs gencode-gencode \
  --stage1-config configs/stage1.nocache.yaml \
  --max-parallel-trains 4 \
  --max-parallel-cells 4 \
  --max-parallel-tests 4 \
  --total-cpus 64
```

Useful slices:


| Goal                                                     | Extra args                                                                                       |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Same-annotation only (gencode↔gencode, refseq↔refseq, …) | `--only-same-annotation`                                                                         |
| Specific pairs                                           | `--annotation-pairs gencode-refseq gencode-ensembl`                                              |
| Full cross grid (large)                                  | omit `--annotation-pairs` (default excludes same-annotation diagonal unless flags say otherwise) |


Primary outputs:

```text
runs/cross_annotation_repro/
  _cross_annotation_shared_train/<modality>__train_<anno>/models/
  <modality>__train_<anno>__test_<anno>/reports/benchmark_summary.csv
```

### Other paper experiments

```bash
telos-repro list          # ids + linked plots
telos-repro status        # whether local summaries exist

telos-repro run novel_phase_a_cross_annotation -- …
telos-repro run human_gencode_feature_window -- …
telos-repro run stage1_feature_importance_gencode -- …
# mouse / tissue / refseq-novel: see notes from `telos-repro list -v`
```

Stage I settings: `configs/stage1.defaults.yaml` (or `configs/stage1.nocache.yaml`
with `cache_dir: null`). That YAML is passed into product Telos as each job’s config.

---

## 4. Make figures

After summaries exist:

```bash
# Cross-annotation AUPR bar PDFs
telos-repro plot cross_annotation -- --root runs/cross_annotation_repro

# Or regenerate everything registered
telos-repro plot all
```


| Plot id               | Typical source run                       |
| --------------------- | ---------------------------------------- |
| `cross_annotation`    | `runs/cross_annotation_repro`            |
| `novel_phase_a_cross` | `runs/novel_phase_a_cross_annotation`    |
| `mouse`               | `runs/mouse_cross_species_gencode`       |
| `tissue`              | `runs/tissue_human_gencode`              |
| `window`              | `runs/human_gencode_feature_window`      |
| `feature_importance`  | `runs/stage1_feature_importance_gencode` |
| `refseq_novel`        | `runs/refseq_novel_eval`                 |


PDFs go under `plot_v2/` (or `figures_root` from paths). Full map: `[figures/CATALOG.md](figures/CATALOG.md)`.

Figure-driving numbers live in `benchmark_summary.csv`; PDF pixels need not match a prior run byte-for-byte.

---

## How Telos fits in


| Layer               | Responsibility                                                         |
| ------------------- | ---------------------------------------------------------------------- |
| **Telos** (product) | `train` / `predict`, Stage I–II features & models                      |
| **This repo**       | Bundle matrix, shared-train grid, PR evaluation, experiment CLI, plots |


You do not need a second train stack. Orchestration calls Telos through
`telos_repro.backend` (see `[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)`).

---

## Layout

```text
src/telos_repro/     CLI, backend → Telos, benchmark, evaluation, plotting
src/experiments/     Paper experiment entrypoints
configs/             paths template, stage1, experiment registry
workflow/            FASTQ → bundles (Snakemake)
figures/             Figure catalog
docs/                DATA, ARCHITECTURE, …
```

---

## More documentation

- `[docs/DATA.md](docs/DATA.md)` — accessions, Snakemake, bundle layout  

