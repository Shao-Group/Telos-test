#!/usr/bin/env bash
# Clean telos_v2 Tier-1 cell with isolated Stage I cache (no Telos feature contamination).
set -euo pipefail

REPO_ROOT="/datadisk1/ixk5174/project_repo/Telos-repro"
OUTDIR="${REPO_ROOT}/runs/parity_tier1_telos_v2_clean/sr__train_gencode__test_gencode"
LOGDIR="${REPO_ROOT}/runs/parity_tier1_telos_v2_clean/logs"
CACHE="${REPO_ROOT}/runs/telos_stage1_feature_cache_v2_clean"
mkdir -p "${OUTDIR}" "${LOGDIR}" "${CACHE}"

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate irtesam-berth

export TELOS_REPRO_BACKEND=telos_v2
export TELOS_BUNDLES_ROOT="/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles"
export TELOS_STAGE1_CACHE_DIR="${CACHE}"
export GFFCOMPARE="/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/clean_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"

echo "[v2-clean] start ${STAMP} cache=${CACHE}" | tee -a "${LOG}"

set +e
python -m telos_v2.cli benchmark \
  --config "${REPO_ROOT}/configs/parity_tier1.example.yaml" \
  --outdir "${OUTDIR}" \
  2>&1 | tee -a "${LOG}"
BENCH_RC=${PIPESTATUS[0]}
set -e
echo "[v2-clean] benchmark exit=${BENCH_RC}" | tee -a "${LOG}"

SUMMARY="${OUTDIR}/reports/benchmark_summary.csv"
set +e
python -m telos_repro parity 1 \
  --backend telos_v2 \
  --local-summary "${SUMMARY}" \
  --report "${REPO_ROOT}/runs/parity_reports/tier1_telos_v2_clean_${STAMP}.json" \
  2>&1 | tee -a "${LOG}"
PARITY_RC=${PIPESTATUS[0]}
set -e

# cache column check
python - <<PY 2>&1 | tee -a "${LOG}"
from pathlib import Path
import pandas as pd
root=Path("${CACHE}")
subs=sorted(p for p in root.iterdir() if p.is_dir())
print("cache_dirs", [p.name for p in subs])
if subs:
    df=pd.read_pickle(subs[0]/"df_all.pkl")
    print("has up_down_stream_ratio", "up_down_stream_ratio" in df.columns)
    print("has upstream_downstream_ratio", "upstream_downstream_ratio" in df.columns)
PY

echo "[v2-clean] parity exit=${PARITY_RC}" | tee -a "${LOG}"
echo "[v2-clean] done" | tee -a "${LOG}"
exit "${PARITY_RC}"
