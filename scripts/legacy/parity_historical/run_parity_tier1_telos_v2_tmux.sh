#!/usr/bin/env bash
# Tier-1 A-B: same cell with telos_v2 backend for drift isolation.
set -euo pipefail

REPO_ROOT="/datadisk1/ixk5174/project_repo/Telos-repro"
OUTDIR="${REPO_ROOT}/runs/parity_tier1_telos_v2/sr__train_gencode__test_gencode"
LOGDIR="${REPO_ROOT}/runs/parity_tier1_telos_v2/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}" "${REPO_ROOT}/runs/telos_stage1_feature_cache"

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate irtesam-berth

export TELOS_REPRO_BACKEND=telos_v2
export TELOS_BUNDLES_ROOT="/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles"
export TELOS_STAGE1_CACHE_DIR="${REPO_ROOT}/runs/telos_stage1_feature_cache"
export GFFCOMPARE="/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/tier1_v2_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/tier1_v2_latest.log"

echo "[parity-tier1-v2] start ${STAMP}" | tee -a "${LOG}"
echo "[parity-tier1-v2] backend=${TELOS_REPRO_BACKEND}" | tee -a "${LOG}"

set +e
python -m telos_v2.cli benchmark \
  --config "${REPO_ROOT}/configs/parity_tier1.example.yaml" \
  --outdir "${OUTDIR}" \
  2>&1 | tee -a "${LOG}"
BENCH_RC=${PIPESTATUS[0]}
set -e
echo "[parity-tier1-v2] benchmark exit=${BENCH_RC}" | tee -a "${LOG}"

SUMMARY="${OUTDIR}/reports/benchmark_summary.csv"
set +e
python -m telos_repro parity 1 \
  --backend telos_v2 \
  --local-summary "${SUMMARY}" \
  --report "${REPO_ROOT}/runs/parity_reports/tier1_telos_v2_${STAMP}.json" \
  2>&1 | tee -a "${LOG}"
PARITY_RC=${PIPESTATUS[0]}
set -e

echo "[parity-tier1-v2] parity exit=${PARITY_RC}" | tee -a "${LOG}"
echo "[parity-tier1-v2] done" | tee -a "${LOG}"
exit "${PARITY_RC}"
