#!/usr/bin/env bash
# Tier-1 smoke / parity cell: product telos backend.
# Requires bundles + shared-train models (see configs/parity_tier1.example.yaml).
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

YAML="${REPO_ROOT}/configs/parity_tier1.example.yaml"
OUTDIR="${REPO_ROOT}/runs/parity_tier1_golden/telos"
CACHE="${REPO_ROOT}/runs/telos_stage1_feature_cache_golden_telos"
LOGDIR="${REPO_ROOT}/runs/parity_tier1_golden/logs"

mkdir -p "${OUTDIR}" "${LOGDIR}" "${CACHE}" "${REPO_ROOT}/runs/parity_reports"
export TELOS_STAGE1_CACHE_DIR="${CACHE}"
export TELOS_BUNDLES_ROOT
export GFFCOMPARE

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/tier1_golden_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"

echo "[tier1] start ${STAMP}" | tee -a "${LOG}"
echo "[tier1] outdir=${OUTDIR}" | tee -a "${LOG}"
echo "[tier1] bundles=${TELOS_BUNDLES_ROOT}" | tee -a "${LOG}"
echo "[tier1] telos_src=${TELOS_SRC}" | tee -a "${LOG}"
echo "[tier1] gffcompare=${GFFCOMPARE}" | tee -a "${LOG}"

python - <<'PY' 2>&1 | tee -a "${LOG}"
import importlib
for m in ("telos_v2", "telos_v2.commands.train", "telos_v2.pipeline_core"):
    try:
        importlib.import_module(m)
        raise SystemExit(f"FAIL: {m} still importable")
    except ModuleNotFoundError:
        print("confirmed gone:", m)
from telos_repro.backend import get_backend_name, resolve_backend
assert get_backend_name() == "telos"
assert resolve_backend().name == "telos"
print("backend ok: telos")
PY

rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"

set +e
python -m telos_repro.pipeline_cli benchmark \
  --config "${YAML}" \
  --outdir "${OUTDIR}" \
  2>&1 | tee -a "${LOG}"
BENCH_RC=${PIPESTATUS[0]}
set -e
echo "[tier1] bench exit=${BENCH_RC}" | tee -a "${LOG}"

SUMMARY="${OUTDIR}/reports/benchmark_summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "[tier1] ERROR: missing ${SUMMARY}" | tee -a "${LOG}"
  exit 1
fi

# Optional: compare to goldens if goldens_runs_root / REF_RUNS_ROOT is configured.
if [[ -n "${REF_RUNS_ROOT}" ]]; then
  set +e
  python -m telos_repro parity 1 \
    --backend telos \
    --local-summary "${SUMMARY}" \
    --report "${REPO_ROOT}/runs/parity_reports/tier1_golden_telos_${STAMP}.json" \
    2>&1 | tee -a "${LOG}"
  PARITY_RC=${PIPESTATUS[0]}
  set -e
  echo "[tier1] parity exit=${PARITY_RC}" | tee -a "${LOG}"
  exit "${PARITY_RC}"
fi

echo "[tier1] done (no REF_RUNS_ROOT; skipped golden parity compare)" | tee -a "${LOG}"
exit "${BENCH_RC}"
