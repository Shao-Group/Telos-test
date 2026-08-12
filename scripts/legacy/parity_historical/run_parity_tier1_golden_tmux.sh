#!/usr/bin/env bash
# Tier-1 vs FROZEN golden (exact paper results), both backends, isolated caches.
# Restores Stage I up_down_stream_ratio quirk to match frozen shared-train models.
set -euo pipefail

REPO_ROOT="/datadisk1/ixk5174/project_repo/Telos-repro"
TELOS_SRC="/datadisk1/ixk5174/project_repo/Telos/src"
YAML="${REPO_ROOT}/configs/parity_tier1.example.yaml"
LOGDIR="${REPO_ROOT}/runs/parity_tier1_golden/logs"
GFFCOMPARE="/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"

mkdir -p "${LOGDIR}" "${REPO_ROOT}/runs/parity_reports"

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate irtesam-berth

export GFFCOMPARE
export TELOS_BUNDLES_ROOT="/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles"
export PYTHONPATH="${REPO_ROOT}/src:${TELOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/golden_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"
echo "[golden] start ${STAMP}" | tee -a "${LOG}"

run_one() {
  local backend="$1"
  local outdir="${REPO_ROOT}/runs/parity_tier1_golden/${backend}"
  local cache="${REPO_ROOT}/runs/telos_stage1_feature_cache_golden_${backend}"
  rm -rf "${outdir}"
  mkdir -p "${outdir}" "${cache}"
  export TELOS_REPRO_BACKEND="${backend}"
  export TELOS_STAGE1_CACHE_DIR="${cache}"
  echo "[golden] bench ${backend} cache=${cache}" | tee -a "${LOG}"
  set +e
  python -m telos_v2.cli benchmark --config "${YAML}" --outdir "${outdir}" 2>&1 | tee -a "${LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[golden] ${backend} bench exit=${rc}" | tee -a "${LOG}"
  set +e
  python -m telos_repro parity 1 \
    --backend "${backend}" \
    --local-summary "${outdir}/reports/benchmark_summary.csv" \
    --report "${REPO_ROOT}/runs/parity_reports/tier1_golden_${backend}_${STAMP}.json" \
    2>&1 | tee -a "${LOG}"
  local prc=${PIPESTATUS[0]}
  set -e
  echo "[golden] ${backend} parity exit=${prc}" | tee -a "${LOG}"
  return "${prc}"
}

V2_RC=0
TELOS_RC=0
run_one telos_v2 || V2_RC=$?
run_one telos || TELOS_RC=$?

python - <<PY 2>&1 | tee -a "${LOG}"
from pathlib import Path
import pandas as pd, json
repo = Path("${REPO_ROOT}")
stamp = "${STAMP}"
out = {
  "tier": "1_golden_both",
  "stamp": stamp,
  "note": "parity gate = frozen Telos-test golden, not retrain A-B",
  "telos_v2_parity_rc": int("${V2_RC}"),
  "telos_parity_rc": int("${TELOS_RC}"),
}
for backend in ("telos_v2", "telos"):
  cache = repo / f"runs/telos_stage1_feature_cache_golden_{backend}"
  subs = sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []
  info = {"cache_dirs": [p.name for p in subs]}
  if subs:
    df = pd.read_pickle(subs[0] / "df_all.pkl")
    info["has_up_down_stream_ratio"] = "up_down_stream_ratio" in df.columns
    info["has_upstream_downstream_ratio"] = "upstream_downstream_ratio" in df.columns
    if "up_down_stream_ratio" in df.columns:
      info["up_down_nonzero_frac"] = float((df["up_down_stream_ratio"].fillna(0) != 0).mean())
  out[backend] = info
  rp = repo / f"runs/parity_reports/tier1_golden_{backend}_{stamp}.json"
  if rp.is_file():
    out[f"{backend}_report"] = str(rp)
path = repo / f"runs/parity_reports/tier1_golden_both_{stamp}.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2))
print("wrote", path)
raise SystemExit(0 if int("${V2_RC}") == 0 and int("${TELOS_RC}") == 0 else 1)
PY
FINAL=$?
echo "[golden] done final=${FINAL}" | tee -a "${LOG}"
exit "${FINAL}"
