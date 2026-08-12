#!/usr/bin/env bash
# Launch 3 parallel long-read rnaseq bundle runs in tmux:
#   cdna, drna, pacbio
#
# Reproducibility defaults:
# - REGENERATE=1 (delete existing per-sample work dir before rerun)
# - SKIP_EXISTING unset
# - logs overwritten each run (no new log file names)
#
# Usage:
#   ./scripts/run_long_reads_tmux.sh
#   ANNOTATION_PROFILE=refseq ./scripts/run_long_reads_tmux.sh
#   ISOQUANT_CONDA_ENV=isoquant ISOQUANT_SCRIPT=isoquant.py ANNOTATION_PROFILE=refseq ./scripts/run_long_reads_tmux.sh
#   TMUX_SESSION=rnaseq_refseq_long THREADS=20 THREADS_ISO=40 ./scripts/run_long_reads_tmux.sh

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

TOTAL_THREADS="$(nproc)"
THREADS="${THREADS:-$(( TOTAL_THREADS / 3 ))}"
if (( THREADS < 1 )); then THREADS=1; fi
THREADS_ISO="${THREADS_ISO:-$(( THREADS * 2 ))}"
if (( THREADS_ISO < 1 )); then THREADS_ISO=1; fi

TMUX_SESSION="${TMUX_SESSION:-rnaseq_${ANNOTATION_PROFILE}_long_regen}"
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
THREADS="${THREADS}" \
THREADS_ISO="${THREADS_ISO}" \
REGENERATE=1 \
SKIP_EXISTING= \
bash "${RUNNER}" > "tmp/run_${ANNOTATION_PROFILE}_${only}.log" 2>&1
EOF
}

mkdir -p "${REPO_ROOT}/tmp"

tmux new-session -d -s "${TMUX_SESSION}" -n cdna "$(mk_cmd cdna)"
tmux new-window -t "${TMUX_SESSION}" -n drna "$(mk_cmd drna)"
tmux new-window -t "${TMUX_SESSION}" -n pacbio "$(mk_cmd pacbio)"

echo "Started tmux session: ${TMUX_SESSION}"
echo "  annotation:   ${ANNOTATION_PROFILE}"
echo "  isoquant env:    ${ISOQUANT_CONDA_ENV:-<same as CONDA_ENV>}"
echo "  isoquant script: ${ISOQUANT_SCRIPT:-isoquant}"
echo "  threads:      ${THREADS}"
echo "  threads_iso:  ${THREADS_ISO}"
echo "Logs (overwritten):"
echo "  tmp/run_${ANNOTATION_PROFILE}_cdna.log"
echo "  tmp/run_${ANNOTATION_PROFILE}_drna.log"
echo "  tmp/run_${ANNOTATION_PROFILE}_pacbio.log"
echo "Attach with:"
echo "  tmux attach -t ${TMUX_SESSION}"
