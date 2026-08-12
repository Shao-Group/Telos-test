#!/usr/bin/env bash
# Launch 4 parallel rnaseq bundle runs in tmux:
#   sr, cdna, drna, pacbio
#
# This wrapper enforces reproducibility defaults:
# - REGENERATE=1 (delete existing work dir before each sample run)
# - SKIP_EXISTING unset
#
# Usage:
#   ./scripts/run_all_fastq_bundles_tmux.sh
#   ANNOTATION_PROFILE=gencode ./scripts/run_all_fastq_bundles_tmux.sh
#   THREADS=6 THREADS_ISO=16 ./scripts/run_all_fastq_bundles_tmux.sh
#   TMUX_SESSION=rnaseq_ens ANNOTATION_PROFILE=ensembl ./scripts/run_all_fastq_bundles_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required but not found on PATH." >&2
  exit 1
fi

ANNOTATION_PROFILE="${ANNOTATION_PROFILE:-gencode}"
CONDA_ENV="${CONDA_ENV:-irtesam-berth}"
ISOQUANT_CONDA_ENV="${ISOQUANT_CONDA_ENV:-}"
ISOQUANT_SCRIPT="${ISOQUANT_SCRIPT:-}"
HISAT2_STRANDNESS="${HISAT2_STRANDNESS-RF}"

TOTAL_THREADS="$(nproc)"
THREADS="${THREADS:-$(( TOTAL_THREADS / 4 ))}"
if (( THREADS < 1 )); then THREADS=1; fi
THREADS_ISO="${THREADS_ISO:-$(( THREADS * 2 ))}"
if (( THREADS_ISO < 1 )); then THREADS_ISO=1; fi

TMUX_SESSION="${TMUX_SESSION:-rnaseq_${ANNOTATION_PROFILE}_regen}"
RUNNER="${REPO_ROOT}/scripts/run_all_fastq_bundles.sh"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${TMUX_SESSION}" >&2
  echo "Use TMUX_SESSION=<new_name> or kill old session first." >&2
  exit 1
fi

mk_cmd() {
  local only="$1"
  cat <<EOF
cd "${REPO_ROOT}" && \
ANNOTATION_PROFILE="${ANNOTATION_PROFILE}" \
ONLY="${only}" \
CONDA_ENV="${CONDA_ENV}" \
ISOQUANT_CONDA_ENV="${ISOQUANT_CONDA_ENV}" \
ISOQUANT_SCRIPT="${ISOQUANT_SCRIPT}" \
HISAT2_STRANDNESS="${HISAT2_STRANDNESS}" \
THREADS="${THREADS}" \
THREADS_ISO="${THREADS_ISO}" \
REGENERATE=1 \
SKIP_EXISTING= \
bash "${RUNNER}" 2>&1 | tee "tmp/run_${ANNOTATION_PROFILE}_${only}.log"
EOF
}

mkdir -p "${REPO_ROOT}/tmp"

tmux new-session -d -s "${TMUX_SESSION}" -n sr "$(mk_cmd sr)"
tmux new-window -t "${TMUX_SESSION}" -n cdna "$(mk_cmd cdna)"
tmux new-window -t "${TMUX_SESSION}" -n drna "$(mk_cmd drna)"
tmux new-window -t "${TMUX_SESSION}" -n pacbio "$(mk_cmd pacbio)"

echo "Started tmux session: ${TMUX_SESSION}"
echo "  annotation:   ${ANNOTATION_PROFILE}"
echo "  threads:      ${THREADS}"
echo "  threads_iso:  ${THREADS_ISO}"
echo "Attach with:"
echo "  tmux attach -t ${TMUX_SESSION}"
