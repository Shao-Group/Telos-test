#!/usr/bin/env bash
# Launch four parallel mouse rnaseq_pipeline runs in one tmux session:
#   cdna (nanopore), drna (nanopore + dRNA), pacbio, sr (short paired, unstranded).
#
# Prerequisites (one-time):
#   - Decompressed reference: genome/mouse/GRCm39.primary_assembly.genome.fa
#   - Decompressed GTF:      genome/mouse/gencode.vM38.primary_assembly.basic.annotation.gtf
#   - HISAT2 index prefix:  genome/mouse/hisat2_grcm39/grcm39  (same path passed to hisat2 -x)
#   - fastq/mouse/sr:        mouse_R1.fastq.gz / mouse_R2.fastq.gz → ENCFF pair
#
# Usage:
#   ./scripts/run_mouse_rnaseq_parallel_tmux.sh
#   TMUX_SESSION=mouse_rnaseq_v1 THREADS=6 THREADS_ISO=24 ./scripts/run_mouse_rnaseq_parallel_tmux.sh
#   MOUSE_GENOME_DIR=/path/to/genome/mouse MOUSE_RESULTS_DIR=/path/to/out ./scripts/run_mouse_rnaseq_parallel_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required but not found on PATH." >&2
  exit 1
fi

MOUSE_GENOME_DIR="${MOUSE_GENOME_DIR:-${REPO_ROOT}/genome/mouse}"
MOUSE_FASTQ_DIR="${MOUSE_FASTQ_DIR:-${REPO_ROOT}/fastq/mouse}"
MOUSE_RESULTS_DIR="${MOUSE_RESULTS_DIR:-${REPO_ROOT}/results/mouse}"

REFA="${REFA:-${MOUSE_GENOME_DIR}/GRCm39.primary_assembly.genome.fa}"
REFG="${REFG:-${MOUSE_GENOME_DIR}/gencode.vM38.primary_assembly.basic.annotation.gtf}"
HISAT2_INDEX="${HISAT2_INDEX:-${MOUSE_GENOME_DIR}/hisat2_grcm39/grcm39}"

CONDA_ENV="${CONDA_ENV:-irtesam-berth}"
ISOQUANT_CONDA_ENV="${ISOQUANT_CONDA_ENV:-isoquant}"
ISOQUANT_SCRIPT="${ISOQUANT_SCRIPT:-isoquant.py}"

TOTAL_THREADS="$(nproc)"
THREADS="${THREADS:-$(( TOTAL_THREADS / 4 ))}"
if (( THREADS < 1 )); then THREADS=1; fi
THREADS_ISO="${THREADS_ISO:-$(( THREADS * 2 ))}"
if (( THREADS_ISO < 1 )); then THREADS_ISO=1; fi

TMUX_SESSION="${TMUX_SESSION:-mouse_rnaseq_pipeline}"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    echo "Decompress genome/mouse *.gz if needed, then re-run." >&2
    exit 1
  fi
}

require_file "Reference FASTA" "${REFA}"
require_file "Reference GTF" "${REFG}"

if [[ ! -f "${HISAT2_INDEX}.1.ht2" ]] && [[ ! -f "${HISAT2_INDEX}.1.ht2l" ]]; then
  echo "ERROR: HISAT2 index not found at prefix: ${HISAT2_INDEX}" >&2
  echo "Build with: python3 -m rnaseq_pipeline build-hisat2-index ..." >&2
  exit 1
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${TMUX_SESSION}" >&2
  echo "Use TMUX_SESSION=<new_name> or: tmux kill-session -t ${TMUX_SESSION}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/tmp" "${MOUSE_RESULTS_DIR}"

mk_cmd_cdna() {
  cat <<EOF
cd "${REPO_ROOT}" && \
export PYTHONPATH="${REPO_ROOT}/src" && \
python3 -m rnaseq_pipeline run \
  --preset nanopore \
  --fastq "${MOUSE_FASTQ_DIR}/cdna" \
  --work-dir "${MOUSE_RESULTS_DIR}/cdna" \
  --ref-fasta "${REFA}" \
  --ref-gtf "${REFG}" \
  --conda-env "${CONDA_ENV}" \
  --isoquant-conda-env "${ISOQUANT_CONDA_ENV}" \
  --isoquant-script "${ISOQUANT_SCRIPT}" \
  --threads-align "${THREADS}" \
  --threads-assembly "${THREADS}" \
  --threads-isoquant "${THREADS_ISO}" \
  2>&1 | tee "${REPO_ROOT}/tmp/run_mouse_cdna.log"
EOF
}

mk_cmd_drna() {
  cat <<EOF
cd "${REPO_ROOT}" && \
export PYTHONPATH="${REPO_ROOT}/src" && \
python3 -m rnaseq_pipeline run \
  --preset nanopore \
  --nanopore-library drna \
  --fastq "${MOUSE_FASTQ_DIR}/drna" \
  --work-dir "${MOUSE_RESULTS_DIR}/drna" \
  --ref-fasta "${REFA}" \
  --ref-gtf "${REFG}" \
  --conda-env "${CONDA_ENV}" \
  --isoquant-conda-env "${ISOQUANT_CONDA_ENV}" \
  --isoquant-script "${ISOQUANT_SCRIPT}" \
  --threads-align "${THREADS}" \
  --threads-assembly "${THREADS}" \
  --threads-isoquant "${THREADS_ISO}" \
  2>&1 | tee "${REPO_ROOT}/tmp/run_mouse_drna.log"
EOF
}

mk_cmd_pacbio() {
  cat <<EOF
cd "${REPO_ROOT}" && \
export PYTHONPATH="${REPO_ROOT}/src" && \
python3 -m rnaseq_pipeline run \
  --preset pacbio \
  --fastq "${MOUSE_FASTQ_DIR}/pacbio" \
  --work-dir "${MOUSE_RESULTS_DIR}/pacbio" \
  --ref-fasta "${REFA}" \
  --ref-gtf "${REFG}" \
  --conda-env "${CONDA_ENV}" \
  --isoquant-conda-env "${ISOQUANT_CONDA_ENV}" \
  --isoquant-script "${ISOQUANT_SCRIPT}" \
  --threads-align "${THREADS}" \
  --threads-assembly "${THREADS}" \
  --threads-isoquant "${THREADS_ISO}" \
  2>&1 | tee "${REPO_ROOT}/tmp/run_mouse_pacbio.log"
EOF
}

mk_cmd_sr() {
  cat <<EOF
cd "${REPO_ROOT}" && \
export PYTHONPATH="${REPO_ROOT}/src" && \
python3 -m rnaseq_pipeline run \
  --preset short_paired \
  --fastq "${MOUSE_FASTQ_DIR}/sr" \
  --work-dir "${MOUSE_RESULTS_DIR}/sr" \
  --ref-fasta "${REFA}" \
  --ref-gtf "${REFG}" \
  --hisat2-index "${HISAT2_INDEX}" \
  --conda-env "${CONDA_ENV}" \
  --threads-align "${THREADS}" \
  --threads-assembly "${THREADS}" \
  2>&1 | tee "${REPO_ROOT}/tmp/run_mouse_sr.log"
EOF
}

tmux new-session -d -s "${TMUX_SESSION}" -n cdna "$(mk_cmd_cdna)"
tmux new-window -t "${TMUX_SESSION}" -n drna "$(mk_cmd_drna)"
tmux new-window -t "${TMUX_SESSION}" -n pacbio "$(mk_cmd_pacbio)"
tmux new-window -t "${TMUX_SESSION}" -n sr "$(mk_cmd_sr)"

echo "Started tmux session: ${TMUX_SESSION}"
echo "  genome dir:     ${MOUSE_GENOME_DIR}"
echo "  fastq dir:      ${MOUSE_FASTQ_DIR}"
echo "  results dir:    ${MOUSE_RESULTS_DIR}"
echo "  conda env:      ${CONDA_ENV}"
echo "  isoquant env:   ${ISOQUANT_CONDA_ENV} (${ISOQUANT_SCRIPT})"
echo "  threads align/assembly: ${THREADS}"
echo "  threads isoquant:       ${THREADS_ISO}"
echo "Logs:"
echo "  ${REPO_ROOT}/tmp/run_mouse_cdna.log"
echo "  ${REPO_ROOT}/tmp/run_mouse_drna.log"
echo "  ${REPO_ROOT}/tmp/run_mouse_pacbio.log"
echo "  ${REPO_ROOT}/tmp/run_mouse_sr.log"
echo "Attach with:"
echo "  tmux attach -t ${TMUX_SESSION}"
