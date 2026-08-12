#!/usr/bin/env bash
# Multi-dataset parity: optional frozen-model phase + retrain phase for a few SR cells.
# Requires TELOS_BUNDLES_ROOT and (for Phase A) REF_RUNS_ROOT pointing at a reference runs tree.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

if [[ -z "${REF_RUNS_ROOT}" ]]; then
  echo "ERROR: set REF_RUNS_ROOT (or goldens_runs_root in paths.yaml) for multi-dataset parity." >&2
  exit 2
fi

DATA_CP="${TELOS_BUNDLES_ROOT}"
GOLDEN_CSV="${REF_RUNS_ROOT}/cross_annotation_repro/sr__train_gencode__test_gencode/reports/benchmark_summary.csv"
FROZEN_MODELS="${REF_RUNS_ROOT}/cross_annotation_repro/_cross_annotation_shared_train/sr__train_gencode/models"
FROZEN_TESTS="${REF_RUNS_ROOT}/cross_annotation_repro/sr__train_gencode__test_gencode/tests"
STAGE1="${REPO_ROOT}/configs/stage1.defaults.yaml"
REF_GTF="${REF_GTF:-${REPO_ROOT}/genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf}"

OUT_A="${REPO_ROOT}/runs/parity_multi/phaseA_frozen_models"
OUT_B="${REPO_ROOT}/runs/parity_multi/phaseB_retrain"
TRAIN_OUT="${REPO_ROOT}/runs/parity_multi/retrain_sr_gencode"
CACHE_A="${REPO_ROOT}/runs/telos_stage1_feature_cache_multi_a"
CACHE_B="${REPO_ROOT}/runs/telos_stage1_feature_cache_multi_b"
LOGDIR="${REPO_ROOT}/runs/parity_multi/logs"
YAML_A="${REPO_ROOT}/runs/parity_multi/phaseA.yaml"
YAML_B="${REPO_ROOT}/runs/parity_multi/phaseB.yaml"

# Few held-out samples × both assemblers (6 cells).
SAMPLES=(SRR307911 SRR315323 SRR315334)
ASSEMBLERS=(stringtie scallop2)

mkdir -p "${OUT_A}" "${OUT_B}" "${TRAIN_OUT}" "${CACHE_A}" "${CACHE_B}" "${LOGDIR}" \
  "${REPO_ROOT}/runs/parity_reports"

export TELOS_BUNDLES_ROOT="${DATA_CP}"
export GFFCOMPARE

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOGDIR}/multi_${STAMP}.log"
ln -sfn "$(basename "${LOG}")" "${LOGDIR}/latest.log"
echo "[multi] start ${STAMP}" | tee -a "${LOG}"

# --- write YAMLs with live data-cp test paths (manifests still point at missing data/) ---
python - <<PY 2>&1 | tee -a "${LOG}"
from pathlib import Path
import yaml

repo = Path("${REPO_ROOT}")
data_cp = Path("${DATA_CP}")
stage1 = Path("${STAGE1}")
ref_gtf = Path("${REF_GTF}")
samples = ["SRR307911", "SRR315323", "SRR315334"]
assemblers = ["stringtie", "scallop2"]

def tests():
    rows = []
    for sid in samples:
        base = data_cp / "GRCh38_gencode49/sr" / sid
        bam = base / "align/aln.sorted.bam"
        assert bam.is_file(), bam
        for aid in assemblers:
            gtf = base / f"{aid}.gtf"
            tmap = base / f"{aid}.{aid}.gtf.tmap"
            assert gtf.is_file() and tmap.is_file(), (gtf, tmap)
            rows.append({
                "id": f"{sid}__{aid}",
                "assembler_id": aid,
                "bam": str(bam),
                "gtf": str(gtf),
                "ref_gtf": str(ref_gtf),
                "tmap": str(tmap),
                "config": str(stage1.resolve()),
            })
    return rows

analysis = {
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
}

# Phase A: frozen models, predict-only
cfg_a = {
    "train": {"mode": "skip", "model_dir": str(Path("${FROZEN_MODELS}").resolve())},
    "tests": tests(),
    "execution": {"stop_on_error": True},
    "analysis": analysis,
}
Path("${YAML_A}").write_text(yaml.safe_dump(cfg_a, sort_keys=False))
print("[multi] wrote", "${YAML_A}", "n_tests=", len(cfg_a["tests"]))

# Phase B: retrain on original data/ train BAM (matches frozen train path), then same tests
train_base = data_orig / "GRCh38_gencode49/sr/SRR307903"
train_bam = train_base / "align/aln.sorted.bam"
train_gtf = train_base / "stringtie.gtf"
train_tmap = train_base / "stringtie.stringtie.gtf.tmap"
pool_gtf = train_base / "scallop2.gtf"
pool_tmap = train_base / "scallop2.scallop2.gtf.tmap"
for p in (train_bam, train_gtf, train_tmap, pool_gtf, pool_tmap):
    assert p.is_file(), p
cfg_b = {
    "train": {
        "mode": "run",
        "outdir": str(Path("${TRAIN_OUT}").resolve()),
        "bam": str(train_bam),
        "gtf": str(train_gtf),
        "ref_gtf": str(ref_gtf),
        "tmap": str(train_tmap),
        "gtf_pool": [str(pool_gtf)],
        "tmap_pool": [str(pool_tmap)],
        "config": str(stage1.resolve()),
    },
    "tests": tests(),
    "execution": {"stop_on_error": True},
    "analysis": analysis,
}
Path("${YAML_B}").write_text(yaml.safe_dump(cfg_b, sort_keys=False))
print("[multi] wrote", "${YAML_B}")
print("[multi] train bam=", train_bam)
PY

run_phase() {
  local name="$1" yaml="$2" outdir="$3" cache="$4"
  echo "[multi] === ${name} ===" | tee -a "${LOG}"
  export TELOS_STAGE1_CACHE_DIR="${cache}"
  rm -rf "${outdir}"
  mkdir -p "${outdir}"
  set +e
  python -m telos_repro.pipeline_cli benchmark --config "${yaml}" --outdir "${outdir}" 2>&1 | tee -a "${LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[multi] ${name} bench exit=${rc}" | tee -a "${LOG}"
  return "${rc}"
}

A_RC=0
B_RC=0
run_phase phaseA_frozen "${YAML_A}" "${OUT_A}" "${CACHE_A}" || A_RC=$?
run_phase phaseB_retrain "${YAML_B}" "${OUT_B}" "${CACHE_B}" || B_RC=$?

export MULTI_PHASE_A_RC="${A_RC}"
export MULTI_PHASE_B_RC="${B_RC}"

python - <<PY 2>&1 | tee -a "${LOG}"
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from telos_repro.parity import compare_summaries, write_report, DEFAULT_METRIC_COLUMNS

repo = Path("${REPO_ROOT}")
golden = Path("${GOLDEN_CSV}")
frozen_tests = Path("${FROZEN_TESTS}")
stamp = Path("${LOGDIR}/latest.log").resolve().name
stamp = stamp.removeprefix("multi_").removesuffix(".log")

def sha16(p: Path) -> str | None:
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

def phase_report(name: str, local_root: Path) -> dict:
    local = local_root / "reports/benchmark_summary.csv"
    cmp_ = compare_summaries(local, golden)
    tsv_cmp = []
    for test_dir in sorted((local_root / "tests").glob("*")):
        tid = test_dir.name
        for backend in ("rf", "xgb"):
            loc = test_dir / "predictions" / f"transcripts.ranked.{backend}.tsv"
            fr = frozen_tests / tid / "predictions" / f"transcripts.ranked.{backend}.tsv"
            ls, fs = sha16(loc), sha16(fr)
            tsv_cmp.append({
                "test_id": tid,
                "backend": backend,
                "local_sha16": ls,
                "frozen_sha16": fs,
                "match": bool(ls and fs and ls == fs),
            })
    return {
        "phase": name,
        "local_summary": str(local),
        "compare": cmp_,
        "ranked_tsv_vs_frozen": tsv_cmp,
        "ranked_tsv_all_match": all(x["match"] for x in tsv_cmp) if tsv_cmp else False,
        "status": "ok" if cmp_["ok"] else "drift",
    }

report = {
    "stamp": stamp,
    "note": "Multi-dataset golden parity: 3 samples × 2 assemblers. "
            "Phase A = frozen models; Phase B = retrain with product Telos then test.",
    "samples": ["SRR307911", "SRR315323", "SRR315334"],
    "assemblers": ["stringtie", "scallop2"],
    "metric_columns": list(DEFAULT_METRIC_COLUMNS),
    "phaseA_bench_rc": int(os.environ.get("MULTI_PHASE_A_RC", "1")),
    "phaseB_bench_rc": int(os.environ.get("MULTI_PHASE_B_RC", "1")),
    "phaseA": phase_report("A_frozen_models", repo / "runs/parity_multi/phaseA_frozen_models"),
    "phaseB": phase_report("B_retrain", repo / "runs/parity_multi/phaseB_retrain"),
}
report["phaseA_ok"] = report["phaseA"]["status"] == "ok" and report["phaseA_bench_rc"] == 0
report["phaseB_ok"] = report["phaseB"]["status"] == "ok" and report["phaseB_bench_rc"] == 0
report["status"] = "ok" if report["phaseA_ok"] and report["phaseB_ok"] else (
    "phaseA_ok_phaseB_drift" if report["phaseA_ok"] else "fail"
)

out = repo / f"runs/parity_reports/multi_dataset_{stamp}.json"
write_report(report, out)
print(json.dumps({
    "status": report["status"],
    "phaseA_ok": report["phaseA_ok"],
    "phaseB_ok": report["phaseB_ok"],
    "phaseA_diffs": report["phaseA"]["compare"]["diffs"],
    "phaseB_diffs": report["phaseB"]["compare"]["diffs"],
    "phaseA_ranked_all_match": report["phaseA"]["ranked_tsv_all_match"],
    "phaseB_ranked_all_match": report["phaseB"]["ranked_tsv_all_match"],
    "report": str(out),
}, indent=2))
raise SystemExit(0 if report["phaseA_ok"] else 1)
PY
FINAL=$?
echo "[multi] done final=${FINAL} status in runs/parity_reports/multi_dataset_*.json" | tee -a "${LOG}"
exit "${FINAL}"
