#!/usr/bin/env bash
# Experiment 2: downstream benchmarks using human GENCODE models + RefSeq-novel eval.
#
# Pipeline:
#   1. mouse_cross_species_human_gencode_train.py
#   2. tissue_human_gencode_train.py
#   3. cross_annotation_repro.py (refseq→gencode only, for RefSeq-trained models)
#   4. evaluate_refseq_novel_cross_annotation.py
#
# Prerequisite: human GENCODE models under HUMAN_ROOT (gencode-gencode shared train).
# Prefer: telos-repro run mouse_cross_species_gencode … / tissue_human_gencode …
#
# Usage:
#   ./scripts/run_exp2.sh
#
# Optional overrides:
#   HUMAN_ROOT=runs/cross_annotation_repro/_cross_annotation_shared_train \
#   REFSEQ_CROSS_OUT=runs/cross-anno-refseq \
#   GFFCOMPARE_BIN=/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare \
#   CONDA_ENV=irtesam-berth \
#   ./scripts/run_exp2.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HUMAN_ROOT="${HUMAN_ROOT:-${REPO_ROOT}/runs/cross_annotation_repro/_cross_annotation_shared_train}"
MOUSE_OUT="${MOUSE_OUT:-${REPO_ROOT}/runs/mouse_cross_species_gencode}"
TISSUE_OUT="${TISSUE_OUT:-${REPO_ROOT}/runs/tissue_human_gencode}"
REFSEQ_CROSS_OUT="${REFSEQ_CROSS_OUT:-${REPO_ROOT}/runs/cross-anno-refseq}"
CROSS_ANNOTATION_ROOT="${CROSS_ANNOTATION_ROOT:-${REFSEQ_CROSS_OUT}}"
REFSEQ_EVAL_OUT="${REFSEQ_EVAL_OUT:-${REPO_ROOT}/runs/refseq_novel_eval/reports}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-irtesam-berth}"

export PYTHONPATH="${REPO_ROOT}/src_v2"

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
  CACHE_DIR="$(python - <<PY
import yaml
from pathlib import Path
p = yaml.safe_load(Path("${PATHS_FILE}").read_text()) or {}
print(p.get("telos_stage1_cache_dir", ""))
PY
)"
  [[ -n "${BUNDLES_ROOT}" ]] && export TELOS_BUNDLES_ROOT="${BUNDLES_ROOT}"
  [[ -n "${CACHE_DIR}" ]] && export TELOS_STAGE1_CACHE_DIR="${CACHE_DIR}"
fi

GFFCOMPARE_BIN="${GFFCOMPARE_BIN:-/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare}"
export GFFCOMPARE="${GFFCOMPARE_BIN}"
if [[ ! -x "${GFFCOMPARE_BIN}" ]]; then
  echo "ERROR: gffcompare not found: ${GFFCOMPARE_BIN}" >&2
  exit 1
fi

if [[ ! -d "${HUMAN_ROOT}" ]]; then
  echo "ERROR: HUMAN_ROOT not found: ${HUMAN_ROOT}" >&2
  echo "Run cross_annotation / Phase A first (or set HUMAN_ROOT)." >&2
  exit 1
fi

echo "[exp2] Step 1/4: mouse cross-species (human GENCODE models)"
python -m experiments.mouse_cross_species_human_gencode_train \
  --human-root "${HUMAN_ROOT}" \
  --outdir "${MOUSE_OUT}"

echo "[exp2] Step 2/4: tissue human GENCODE models"
python -m experiments.tissue_human_gencode_train \
  --human-root "${HUMAN_ROOT}" \
  --outdir "${TISSUE_OUT}"

echo "[exp2] Step 3/4: refseq→gencode cross-annotation (RefSeq-trained models)"
python -m experiments.cross_annotation_repro \
  --outdir "${REFSEQ_CROSS_OUT}" \
  --annotation-pairs refseq-gencode

echo "[exp2] Step 4/4: RefSeq-novel cross-annotation evaluation"
python -m experiments.evaluate_refseq_novel_cross_annotation \
  --root "${CROSS_ANNOTATION_ROOT}" \
  --outdir "${REFSEQ_EVAL_OUT}" \
  --gffcompare-bin "${GFFCOMPARE_BIN}"

echo "[exp2] complete"
echo "  human models:    ${HUMAN_ROOT}"
echo "  mouse outdir:    ${MOUSE_OUT}"
echo "  tissue outdir:   ${TISSUE_OUT}"
echo "  refseq cross:    ${REFSEQ_CROSS_OUT}"
echo "  refseq eval:     ${REFSEQ_EVAL_OUT}"
