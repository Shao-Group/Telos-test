# Figure catalog — paper reproduction

Maps paper-facing figures to `telos-repro` commands and expected outputs.
PDF byte identity is **not** required; figure-driving CSV metrics are what matter.

**Outputs:** under `figures_root` (default `plot_v2/`).

## Quick commands

```bash
telos-repro list
telos-repro plot cross_annotation -- --root runs/cross_annotation_repro
telos-repro plot all
```

Underlying driver: `python -m telos_repro.plotting.plot_experiments …`

## Catalog

| Figure / panel family | Command | Expected outputs |
|----------------------|---------|------------------|
| Cross-annotation AUPR bars | `telos-repro plot cross_annotation` | `plot_v2/cross_annotation_repro/<train>__to__<test>/aupr_bars_*.pdf` |
| Novel Phase A | `telos-repro plot novel_phase_a_cross` | `plot_v2/novel_phase_a_cross_annotation/…` |
| Mouse cross-species | `telos-repro plot mouse` | `plot_v2/mouse_cross_species_gencode/` |
| Tissue | `telos-repro plot tissue` | `plot_v2/tissue_human_gencode/` |
| Feature-window sweep | `telos-repro plot window` | `plot_v2/human_gencode_feature_window/` |
| Stage I feature importance | `telos-repro plot feature_importance` | `plot_v2/stage1_feature_importance_gencode/` |
| RefSeq-novel eval | `telos-repro plot refseq_novel` | `plot_v2/refseq_novel_eval/` |

## Experiment → plot

| Experiment id | Plot id |
|---------------|---------|
| `cross_annotation_repro` | `cross_annotation` |
| `novel_phase_a_cross_annotation` | `novel_phase_a_cross` |
| `mouse_cross_species_gencode` | `mouse` |
| `tissue_human_gencode` | `tissue` |
| `human_gencode_feature_window` | `window` |
| `stage1_feature_importance_gencode` | `feature_importance` |
| `evaluate_refseq_novel` | `refseq_novel` |
