#!/usr/bin/env bash
# Experiment 1: novel-augmented Phase A (gencode-gencode).
#
# Pipeline:
#   1. augment_annotation_with_novel_all_bundles.py
#   2. generate_augmented_tmaps_all_bundles.py
#   3. novel_phase_a_cross_annotation.py
#
# Prefer: telos-repro run novel_phase_a_cross_annotation …
#
# Usage:
#   ./scripts/run_exp1.sh
#
# Optional overrides:
#   NOVEL_REF_OUT=runs/novel_ref_all \
#   PHASE_A_OUT=runs/novel_phase_a_cross_annotation \
#   ANNOTATION_PAIR=gencode-gencode \
#   GFFCOMPARE_BIN=/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare \
#   CONDA_ENV=irtesam-berth \
#   ./scripts/run_exp1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-irtesam-berth}"

export PYTHONPATH="${REPO_ROOT}/src_v2"

# Prefer paths from configs when present
if [[ -f "${REPO_ROOT}/configs/paths.yaml" ]] || [[ -f "${REPO_ROOT}/configs/paths.example.yaml" ]]; then
  PATHS_FILE="${REPO_ROOT}/configs/paths.yaml"
  [[ -f "${PATHS_FILE}" ]] || PATHS_FILE="${REPO_ROOT}/configs/paths.example.yaml"
  GFFCOMPARE_BIN="${GFFCOMPARE_BIN:-$(python - <<PY
import yaml
from pathlib import Path
p = yaml.safe_load(Path("${PATHS_FILE}").read_text()) or {}
print(p.get("gffcompare_bin", ""))
PY
)}"
  BUNDLES_ROOT="$(python - <<PY
import yaml
from pathlib import Path
p = yaml.safe_load(Path("${PATHS_FILE}").read_text()) or {}
print(p.get("bundles_root", ""))
PY
)"
  [[ -n "${BUNDLES_ROOT}" ]] && export TELOS_BUNDLES_ROOT="${BUNDLES_ROOT}"
fi

GFFCOMPARE_BIN="${GFFCOMPARE_BIN:-/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare}"
export GFFCOMPARE="${GFFCOMPARE_BIN}"
if [[ ! -x "${GFFCOMPARE_BIN}" ]]; then
  echo "ERROR: gffcompare not found: ${GFFCOMPARE_BIN}" >&2
  exit 1
fi

NOVEL_REF_OUT="${NOVEL_REF_OUT:-${REPO_ROOT}/runs/novel_ref_all}"
PHASE_A_OUT="${PHASE_A_OUT:-${REPO_ROOT}/runs/novel_phase_a_cross_annotation}"
ANNOTATION_PAIR="${ANNOTATION_PAIR:-gencode-gencode}"

AUGMENTED_INDEX="${NOVEL_REF_OUT}/reports/augmented_refs_index.csv"
AUGMENTED_TMAP_INDEX="${NOVEL_REF_OUT}/reports/augmented_tmaps_index.csv"

echo "[exp1] Step 1/3: augment all bundle references"
# python -m experiments.augment_annotation_with_novel_all_bundles \
#   --out-root "${NOVEL_REF_OUT}"

echo "[exp1] Step 2/3: generate augmented tmaps"
# python -m experiments.generate_augmented_tmaps_all_bundles \
#   --refs-index "${AUGMENTED_INDEX}" \
#   --gffcompare-bin "${GFFCOMPARE_BIN}"

echo "[exp1] Step 3/3: novel Phase A (${ANNOTATION_PAIR})"
python -m experiments.novel_phase_a_cross_annotation \
  --outdir "${PHASE_A_OUT}" \
  --augmented-index "${AUGMENTED_INDEX}" \
  --augmented-tmap-index "${AUGMENTED_TMAP_INDEX}" \
  --annotation-pair "${ANNOTATION_PAIR}"

echo "[exp1] complete"
echo "  augmented refs:  ${AUGMENTED_INDEX}"
echo "  augmented tmaps: ${AUGMENTED_TMAP_INDEX}"
echo "  phase A outdir:  ${PHASE_A_OUT}"
echo "  shared train:    ${PHASE_A_OUT}/_phase_a_shared_train"
