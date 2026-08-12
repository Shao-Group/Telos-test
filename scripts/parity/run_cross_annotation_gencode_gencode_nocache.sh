#!/usr/bin/env bash
# gencode→gencode paper benchmark: fresh Stage I features (NO cache).
# Optionally compare summaries to REF_RUNS_ROOT/cross_annotation_repro.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

BUNDLES="${TELOS_BUNDLES_ROOT}"
STAGE1="${REPO_ROOT}/configs/stage1.nocache.yaml"
OUTDIR="${REPO_ROOT}/runs/cross_annotation_repro_gencode_gencode_nocache"
LOGDIR="${OUTDIR}/logs"
REF_RUNS="${REF_RUNS_ROOT:+${REF_RUNS_ROOT}/cross_annotation_repro}"

mkdir -p "${OUTDIR}" "${LOGDIR}" "${REPO_ROOT}/runs/parity_reports"

# Critical: do not inherit / inject Stage I feature cache.
unset TELOS_STAGE1_CACHE_DIR || true
export TELOS_BUNDLES_ROOT="${BUNDLES}"
export GFFCOMPARE

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/gencode_gencode_nocache_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"

echo "[gencode-gencode] start ${STAMP}" | tee -a "${LOG}"
echo "[gencode-gencode] bundles=${BUNDLES}" | tee -a "${LOG}"
echo "[gencode-gencode] stage1=${STAGE1}" | tee -a "${LOG}"
echo "[gencode-gencode] outdir=${OUTDIR}" | tee -a "${LOG}"
echo "[gencode-gencode] TELOS_STAGE1_CACHE_DIR=${TELOS_STAGE1_CACHE_DIR-<unset>}" | tee -a "${LOG}"
python - <<'PY' 2>&1 | tee -a "${LOG}"
import os, yaml
from pathlib import Path
assert "TELOS_STAGE1_CACHE_DIR" not in os.environ, os.environ.get("TELOS_STAGE1_CACHE_DIR")
cfg = yaml.safe_load(Path("configs/stage1.nocache.yaml").read_text())
assert cfg["stage1"]["feature_extraction"]["cache_dir"] in (None, "null")
print("cache disabled: env unset + yaml cache_dir=null")
PY

# --- Feature QC (fresh extract on pacbio train + sr held-out test) ---
QC_DIR="${OUTDIR}/feature_qc"
mkdir -p "${QC_DIR}"
SKIP_QC=0
if [[ "${SKIP_FEATURE_QC:-0}" == "1" ]]; then
  SKIP_QC=1
elif [[ -f "${QC_DIR}/pacbio_ENCFF450VAU_stringtie.json" && -f "${QC_DIR}/sr_SRR307911_stringtie.json" ]]; then
  if python - <<'PY'
import json
from pathlib import Path
root = Path("runs/cross_annotation_repro_gencode_gencode_nocache/feature_qc")
for stem in ("pacbio_ENCFF450VAU_stringtie", "sr_SRR307911_stringtie"):
    d = json.loads((root / f"{stem}.json").read_text())
    if d.get("status") != "ok":
        raise SystemExit(1)
raise SystemExit(0)
PY
  then
    SKIP_QC=1
    echo "[gencode-gencode] feature_qc reuse existing OK reports" | tee -a "${LOG}"
  fi
fi

if [[ "${SKIP_QC}" -eq 0 ]]; then
  set +e
  python scripts/parity/qc_stage1_features.py \
    --bam "${BUNDLES}/GRCh38_gencode49/pacbio/ENCFF450VAU/align/aln.sorted.bam" \
    --gtf "${BUNDLES}/GRCh38_gencode49/pacbio/ENCFF450VAU/stringtie.gtf" \
    --stage1-config "${STAGE1}" \
    --report "${QC_DIR}/pacbio_ENCFF450VAU_stringtie.json" \
    --dump-parquet "${QC_DIR}/pacbio_ENCFF450VAU_stringtie.df_all.pkl" \
    2>&1 | tee -a "${LOG}"
  QC1=${PIPESTATUS[0]}
  python scripts/parity/qc_stage1_features.py \
    --bam "${BUNDLES}/GRCh38_gencode49/sr/SRR307911/align/aln.sorted.bam" \
    --gtf "${BUNDLES}/GRCh38_gencode49/sr/SRR307911/stringtie.gtf" \
    --stage1-config "${STAGE1}" \
    --report "${QC_DIR}/sr_SRR307911_stringtie.json" \
    --dump-parquet "${QC_DIR}/sr_SRR307911_stringtie.df_all.pkl" \
    2>&1 | tee -a "${LOG}"
  QC2=${PIPESTATUS[0]}
  set -e
  echo "[gencode-gencode] feature_qc pacbio_exit=${QC1} sr_exit=${QC2}" | tee -a "${LOG}"
  if [[ "${QC1}" -ne 0 || "${QC2}" -ne 0 ]]; then
    echo "[gencode-gencode] ERROR: feature QC anomalies — aborting benchmark" | tee -a "${LOG}"
    exit 1
  fi
else
  echo "[gencode-gencode] feature_qc skipped (OK reports present or SKIP_FEATURE_QC=1)" | tee -a "${LOG}"
fi

# --- Benchmark grid: all modalities, gencode→gencode, retrain shared axes ---
set +e
python -m experiments.cross_annotation_repro \
  --outdir "${OUTDIR}" \
  --bundles-root "${BUNDLES}" \
  --stage1-config "${STAGE1}" \
  --annotation-pairs gencode-gencode \
  --max-parallel-trains 4 \
  --max-parallel-cells 4 \
  --max-parallel-tests 4 \
  --total-cpus "${TOTAL_CPUS:-64}" \
  2>&1 | tee -a "${LOG}"
BENCH_RC=${PIPESTATUS[0]}
set -e
echo "[gencode-gencode] bench exit=${BENCH_RC}" | tee -a "${LOG}"

if [[ "${BENCH_RC}" -ne 0 ]]; then
  exit "${BENCH_RC}"
fi

# --- Optional compare to a reference runs tree ---
if [[ -z "${REF_RUNS:-}" || ! -d "${REF_RUNS}" ]]; then
  echo "[gencode-gencode] done (no REF_RUNS_ROOT/cross_annotation_repro; skipped compare)" | tee -a "${LOG}"
  exit 0
fi

set +e
python scripts/parity/compare_to_telos_test_runs.py \
  --local-root "${OUTDIR}" \
  --ref-root "${REF_RUNS}" \
  --pair-glob '*__train_gencode__test_gencode' \
  --report "${REPO_ROOT}/runs/parity_reports/gencode_gencode_nocache_vs_ref_runs_${STAMP}.json" \
  2>&1 | tee -a "${LOG}"
CMP_RC=${PIPESTATUS[0]}
set -e
echo "[gencode-gencode] compare exit=${CMP_RC}" | tee -a "${LOG}"
exit "${CMP_RC}"
