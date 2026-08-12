#!/usr/bin/env bash
# Benchmark-only parity after models already trained under runs/parity_retrain/.
# Uses data-cp paths for the Tier-1 test cell (data/bundles manifests are stale).
set -euo pipefail

REPO_ROOT="/datadisk1/ixk5174/project_repo/Telos-repro"
TELOS_SRC="/datadisk1/ixk5174/project_repo/Telos/src"
MODEL_DIR="${REPO_ROOT}/runs/parity_retrain/sr__train_gencode/models"
YAML="${REPO_ROOT}/runs/parity_retrain/parity_tier1_retrain.yaml"
BENCH_V2="${REPO_ROOT}/runs/parity_tier1_retrain/telos_v2"
BENCH_TELOS="${REPO_ROOT}/runs/parity_tier1_retrain/telos"
CACHE_V2="${REPO_ROOT}/runs/telos_stage1_feature_cache_retrain_v2"
CACHE_TELOS="${REPO_ROOT}/runs/telos_stage1_feature_cache_retrain_telos"
LOGDIR="${REPO_ROOT}/runs/parity_retrain/logs"
GFFCOMPARE="/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"
DATA_CP="/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles"
GENOME_GTF="/datadisk1/ixk5174/project_repo/Telos-test/genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf"

mkdir -p "${BENCH_V2}" "${BENCH_TELOS}" "${CACHE_V2}" "${CACHE_TELOS}" "${LOGDIR}" \
  "${REPO_ROOT}/runs/parity_reports"

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate irtesam-berth

export GFFCOMPARE
export TELOS_BUNDLES_ROOT="${DATA_CP}"
export PYTHONPATH="${REPO_ROOT}/src:${TELOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/bench_parity_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest_bench.log"

# Rewrite YAML with live data-cp paths
python - <<PY
from pathlib import Path
import yaml
repo = Path("${REPO_ROOT}")
model_dir = Path("${MODEL_DIR}")
assert (model_dir / "stage1_tss_rf_model.joblib").is_file(), model_dir
base = Path("${DATA_CP}") / "GRCh38_gencode49/sr/SRR307911"
cfg = {
    "train": {"mode": "skip", "model_dir": str(model_dir.resolve())},
    "tests": [{
        "id": "SRR307911__stringtie",
        "assembler_id": "stringtie",
        "bam": str(base / "align/aln.sorted.bam"),
        "gtf": str(base / "stringtie.gtf"),
        "ref_gtf": "${GENOME_GTF}",
        "tmap": str(base / "stringtie.stringtie.gtf.tmap"),
        "config": str(repo / "configs/stage1.defaults.yaml"),
    }],
    "execution": {"stop_on_error": True},
    "analysis": {
        "enabled": True,
        "benchmark_mode": "minimal",
        "debug": {"keep_pr_work": False},
        "pr_vs_baseline": {
            "enabled": True,
            "measure": "cov",
            "plot": False,
            "filter_validation_chroms": True,
            "chromosomes_file": None,
            "save_pr_tables": True,
            "gffcompare_bin": "${GFFCOMPARE}",
        },
    },
}
for p in (cfg["tests"][0]["bam"], cfg["tests"][0]["gtf"], cfg["tests"][0]["tmap"]):
    assert Path(p).is_file(), p
yaml_path = Path("${YAML}")
yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[bench] wrote", yaml_path)
PY

echo "[bench] start ${STAMP}" | tee -a "${LOG}"

# Fresh outdirs so we don't mix failed empty summaries
rm -rf "${BENCH_V2}" "${BENCH_TELOS}"
mkdir -p "${BENCH_V2}" "${BENCH_TELOS}"

echo "[bench] telos_v2" | tee -a "${LOG}"
export TELOS_REPRO_BACKEND=telos_v2
export TELOS_STAGE1_CACHE_DIR="${CACHE_V2}"
set +e
python -m telos_v2.cli benchmark --config "${YAML}" --outdir "${BENCH_V2}" 2>&1 | tee -a "${LOG}"
V2_RC=${PIPESTATUS[0]}
set -e
echo "[bench] telos_v2 exit=${V2_RC}" | tee -a "${LOG}"

echo "[bench] telos" | tee -a "${LOG}"
export TELOS_REPRO_BACKEND=telos
export TELOS_STAGE1_CACHE_DIR="${CACHE_TELOS}"
set +e
python -m telos_v2.cli benchmark --config "${YAML}" --outdir "${BENCH_TELOS}" 2>&1 | tee -a "${LOG}"
TELOS_RC=${PIPESTATUS[0]}
set -e
echo "[bench] telos exit=${TELOS_RC}" | tee -a "${LOG}"

python - <<'PY' 2>&1 | tee -a "${LOG}"
from pathlib import Path
from telos_repro.parity import compare_summaries, write_report, DEFAULT_METRIC_COLUMNS
import hashlib, json, csv

repo = Path("/datadisk1/ixk5174/project_repo/Telos-repro")
v2 = repo / "runs/parity_tier1_retrain/telos_v2/reports/benchmark_summary.csv"
telos = repo / "runs/parity_tier1_retrain/telos/reports/benchmark_summary.csv"

def row_ok(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, f"empty summary {path}"
    r = rows[0]
    missing = [c for c in DEFAULT_METRIC_COLUMNS if r.get(c) in (None, "", "nan", "None")]
    return {"row": r, "missing_metrics": missing, "status": r.get("status")}

v2_info = row_ok(v2)
telos_info = row_ok(telos)
cmp_ = compare_summaries(telos, v2)
ok = (
    cmp_["ok"]
    and not v2_info["missing_metrics"]
    and not telos_info["missing_metrics"]
    and str(v2_info["status"]).lower() in {"ok", "success", "0", ""}
)
# status column may be 'ok' or missing; prefer metric presence
if v2_info["missing_metrics"] or telos_info["missing_metrics"]:
    ok = False

report = {
    "tier": "1_retrain_ab",
    "local": str(telos),
    "golden": str(v2),
    "note": "telos vs telos_v2 after retrain on upstream_downstream_ratio (data-cp test paths)",
    "compare": cmp_,
    "v2_status": v2_info["status"],
    "telos_status": telos_info["status"],
    "v2_missing_metrics": v2_info["missing_metrics"],
    "telos_missing_metrics": telos_info["missing_metrics"],
    "v2_metrics": {c: v2_info["row"].get(c) for c in DEFAULT_METRIC_COLUMNS},
    "telos_metrics": {c: telos_info["row"].get(c) for c in DEFAULT_METRIC_COLUMNS},
    "status": "ok" if ok else "drift_or_fail",
}
for backend, root in [("telos_v2", "telos_v2"), ("telos", "telos")]:
    p = repo / f"runs/parity_tier1_retrain/{root}/tests/SRR307911__stringtie/predictions/transcripts.ranked.rf.tsv"
    if p.is_file():
        report[f"{backend}_rf_sha16"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
out = repo / "runs/parity_reports/tier1_retrain_ab.json"
write_report(report, out)
print(json.dumps(report, indent=2))
print("wrote", out)
raise SystemExit(0 if ok else 1)
PY
AB_RC=$?
echo "[bench] A-B exit=${AB_RC}" | tee -a "${LOG}"
exit "${AB_RC}"
