#!/usr/bin/env bash
# Retrain sr/gencode shared models on upstream_downstream_ratio schema, then Tier-1
# parity with both backends (isolated caches).
set -euo pipefail

REPO_ROOT="/datadisk1/ixk5174/project_repo/Telos-repro"
TELOS_SRC="/datadisk1/ixk5174/project_repo/Telos/src"
BUNDLES="/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles"
STAGE1="${REPO_ROOT}/configs/stage1.defaults.yaml"
TRAIN_OUT="${REPO_ROOT}/runs/parity_retrain/sr__train_gencode"
BENCH_V2="${REPO_ROOT}/runs/parity_tier1_retrain/telos_v2"
BENCH_TELOS="${REPO_ROOT}/runs/parity_tier1_retrain/telos"
CACHE_V2="${REPO_ROOT}/runs/telos_stage1_feature_cache_retrain_v2"
CACHE_TELOS="${REPO_ROOT}/runs/telos_stage1_feature_cache_retrain_telos"
LOGDIR="${REPO_ROOT}/runs/parity_retrain/logs"
GFFCOMPARE="/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"

mkdir -p "${TRAIN_OUT}" "${BENCH_V2}" "${BENCH_TELOS}" "${CACHE_V2}" "${CACHE_TELOS}" "${LOGDIR}" \
  "${REPO_ROOT}/runs/parity_reports"

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate irtesam-berth

export GFFCOMPARE
export TELOS_BUNDLES_ROOT="${BUNDLES}"
export PYTHONPATH="${REPO_ROOT}/src:${TELOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/retrain_parity_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"

echo "[retrain] start ${STAMP}" | tee -a "${LOG}"

# --- 1) Train new models with telos_v2 (unified feature name) ---
export TELOS_REPRO_BACKEND=telos_v2
export TELOS_STAGE1_CACHE_DIR="${CACHE_V2}"

python - <<'PY' 2>&1 | tee -a "${LOG}"
from pathlib import Path
import yaml
from telos_v2.benchmark.matrix import build_benchmark_yaml_mapping, benchmark_mapping_to_yaml_text
from telos_repro.backend import train
from telos_repro.backend.types import TrainRequest

repo = Path("/datadisk1/ixk5174/project_repo/Telos-repro")
bundles = Path("/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles")
stage1 = repo / "configs/stage1.defaults.yaml"
train_out = repo / "runs/parity_retrain/sr__train_gencode"
mapping = build_benchmark_yaml_mapping(
    data_type="sr",
    train_annotation="gencode",
    test_annotation="gencode",
    bundles_root=bundles,
    stage1_config=stage1,
    train_outdir=train_out,
)
train_obj = mapping["train"]
print("[retrain] train bam=", train_obj["bam"])
print("[retrain] gtf_pool n=", len(train_obj.get("gtf_pool") or []))
code = train(
    TrainRequest(
        bam=Path(train_obj["bam"]),
        gtf=Path(train_obj["gtf"]),
        ref_gtf=Path(train_obj["ref_gtf"]),
        tmap=Path(train_obj["tmap"]),
        gtf_pool=tuple(Path(p) for p in (train_obj.get("gtf_pool") or [])) or None,
        tmap_pool=tuple(Path(p) for p in (train_obj.get("tmap_pool") or [])) or None,
        outdir=train_out,
        config_file=stage1,
    )
)
print("[retrain] train exit=", code)
if code != 0:
    raise SystemExit(code)

# Write Tier-1 YAML with live data-cp paths (data/bundles manifests are often stale).
data_cp = Path("/datadisk1/ixk5174/project_repo/Telos-test/data-cp/bundles")
test_base = data_cp / "GRCh38_gencode49/sr/SRR307911"
cfg = {
    "train": {
        "mode": "skip",
        "model_dir": str((train_out / "models").resolve()),
    },
    "tests": [
        {
            "id": "SRR307911__stringtie",
            "assembler_id": "stringtie",
            "bam": str(test_base / "align/aln.sorted.bam"),
            "gtf": str(test_base / "stringtie.gtf"),
            "ref_gtf": "/datadisk1/ixk5174/project_repo/Telos-test/genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf",
            "tmap": str(test_base / "stringtie.stringtie.gtf.tmap"),
            "config": str(stage1.resolve()),
        }
    ],
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
            "gffcompare_bin": "/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare",
        },
    },
}
for p in (cfg["tests"][0]["bam"], cfg["tests"][0]["gtf"], cfg["tests"][0]["tmap"]):
    assert Path(p).is_file(), f"missing test input: {p}"
yaml_path = repo / "runs/parity_retrain/parity_tier1_retrain.yaml"
yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[retrain] wrote", yaml_path)

# Confirm model feature schema
import joblib, json
bundle = joblib.load(train_out / "models/stage1_tss_rf_model.joblib")
fn = list(bundle["feature_names"])
print("[retrain] has up_down_stream_ratio", "up_down_stream_ratio" in fn)
print("[retrain] has upstream_downstream_ratio", "upstream_downstream_ratio" in fn)
print("[retrain] n_features", len(fn))
PY

# --- 2) Benchmark with telos_v2 ---
echo "[retrain] benchmark telos_v2" | tee -a "${LOG}"
export TELOS_REPRO_BACKEND=telos_v2
export TELOS_STAGE1_CACHE_DIR="${CACHE_V2}"
set +e
python -m telos_v2.cli benchmark \
  --config "${REPO_ROOT}/runs/parity_retrain/parity_tier1_retrain.yaml" \
  --outdir "${BENCH_V2}" \
  2>&1 | tee -a "${LOG}"
V2_RC=${PIPESTATUS[0]}
set -e
echo "[retrain] telos_v2 bench exit=${V2_RC}" | tee -a "${LOG}"

# --- 3) Benchmark with telos (fresh cache) ---
echo "[retrain] benchmark telos" | tee -a "${LOG}"
export TELOS_REPRO_BACKEND=telos
export TELOS_STAGE1_CACHE_DIR="${CACHE_TELOS}"
set +e
python -m telos_v2.cli benchmark \
  --config "${REPO_ROOT}/runs/parity_retrain/parity_tier1_retrain.yaml" \
  --outdir "${BENCH_TELOS}" \
  2>&1 | tee -a "${LOG}"
TELOS_RC=${PIPESTATUS[0]}
set -e
echo "[retrain] telos bench exit=${TELOS_RC}" | tee -a "${LOG}"

# --- 4) Compare telos vs telos_v2 (new golden = v2) ---
python - <<'PY' 2>&1 | tee -a "${LOG}"
from pathlib import Path
from telos_repro.parity import compare_summaries, write_report
import hashlib, json

repo = Path("/datadisk1/ixk5174/project_repo/Telos-repro")
v2 = repo / "runs/parity_tier1_retrain/telos_v2/reports/benchmark_summary.csv"
telos = repo / "runs/parity_tier1_retrain/telos/reports/benchmark_summary.csv"
cmp_ = compare_summaries(telos, v2)
report = {
    "tier": "1_retrain_ab",
    "local": str(telos),
    "golden": str(v2),
    "note": "telos vs telos_v2 after retrain on upstream_downstream_ratio",
    "compare": cmp_,
    "status": "ok" if cmp_["ok"] else "drift",
}
# ranked TSV hashes
for backend, root in [("telos_v2", "telos_v2"), ("telos", "telos")]:
    p = repo / f"runs/parity_tier1_retrain/{root}/tests/SRR307911__stringtie/predictions/transcripts.ranked.rf.tsv"
    if p.is_file():
        report[f"{backend}_rf_sha16"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
out = repo / "runs/parity_reports/tier1_retrain_ab.json"
write_report(report, out)
print(json.dumps(report, indent=2))
print("wrote", out)
raise SystemExit(0 if cmp_["ok"] else 1)
PY
AB_RC=$?

echo "[retrain] A-B exit=${AB_RC}" | tee -a "${LOG}"
echo "[retrain] done" | tee -a "${LOG}"
exit "${AB_RC}"
