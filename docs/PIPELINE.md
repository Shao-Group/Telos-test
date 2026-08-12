# Paper benchmark → figures

After install + bundles (`README.md`, `docs/DATA.md`):

```bash
# Main grid slice (example)
telos-repro run cross_annotation_repro -- \
  --outdir runs/cross_annotation_repro \
  --annotation-pairs gencode-gencode \
  --stage1-config configs/stage1.defaults.yaml

# Or: fresh Stage I extract (no cache), optional compare if REF_RUNS_ROOT is set
bash scripts/parity/run_cross_annotation_gencode_gencode_nocache.sh

# Figures
telos-repro plot cross_annotation -- --root runs/cross_annotation_repro
```

Catalog: [`figures/CATALOG.md`](../figures/CATALOG.md).
