#!/usr/bin/env bash
# Download short-read FASTQ from NCBI SRA for every SRR accession that appears as a
# folder name under ENCODE10_DIR (e.g. /datadisk1/shared/data/encode10/SRR307903).
#
# Requires: sra-toolkit (prefetch, fasterq-dump). Default install:
#   /datadisk1/shared/tools/sratoolkit/sratoolkit.2.10.5-centos_linux64
#
# Usage:
#   ./scripts/download_sra_encode10_fastq.sh
#   OUT_DIR=/path/to/out THREADS=16 ./scripts/download_sra_encode10_fastq.sh
#
# prefetch defaults to --max-size 20G; large runs are skipped unless you raise it, e.g.:
#   PREFETCH_MAX_SIZE=100G ./scripts/download_sra_encode10_fastq.sh
#
# Outputs per run: fasterq-dump names from the .sra basename, e.g. <SRR>.sra_1.fastq / <SRR>.sra_2.fastq
# (paired) or <SRR>.sra.fastq (single); or <SRR>_*.fastq when using accession-only input.
# Optional: set GZIP=1 to gzip FASTQ after dump (needs gzip in PATH).

set -euo pipefail

SRATOOLKIT_BIN="${SRATOOLKIT_BIN:-/datadisk1/shared/tools/sratoolkit/sratoolkit.2.10.5-centos_linux64/bin}"
ENCODE10_DIR="${ENCODE10_DIR:-/datadisk1/shared/data/encode10}"
# Project-relative default: short reads under fastq/sr (keeps long reads in fastq/ separate)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/fastq/sr}"
THREADS="${THREADS:-8}"
GZIP="${GZIP:-0}"
# prefetch -X: max download size (tool default 20G is too small for many bulk RNA-seq SRRs)
PREFETCH_MAX_SIZE="${PREFETCH_MAX_SIZE:-100G}"

if [[ ! -d "${SRATOOLKIT_BIN}" ]]; then
  echo "Error: SRATOOLKIT_BIN not found: ${SRATOOLKIT_BIN}" >&2
  exit 1
fi
export PATH="${SRATOOLKIT_BIN}:${PATH}"

if [[ ! -d "${ENCODE10_DIR}" ]]; then
  echo "Error: ENCODE10_DIR not found: ${ENCODE10_DIR}" >&2
  exit 1
fi

mapfile -t SRR_IDS < <(find "${ENCODE10_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'SRR[0-9]*' -printf '%f\n' | sort -V)
if [[ ${#SRR_IDS[@]} -eq 0 ]]; then
  echo "Error: No SRR* directories under ${ENCODE10_DIR}" >&2
  exit 1
fi

echo "ENCODE10_DIR=${ENCODE10_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "PREFETCH_MAX_SIZE=${PREFETCH_MAX_SIZE}"
echo "SRR runs (${#SRR_IDS[@]}): ${SRR_IDS[*]}"
echo ""

mkdir -p "${OUT_DIR}"

for srr in "${SRR_IDS[@]}"; do
  run_out="${OUT_DIR}/${srr}"
  mkdir -p "${run_out}"

  # Skip if FASTQ already present (single .fastq, paired <SRR>_*.fastq, or <SRR>.sra_* from dump of <SRR>.sra)
  if [[ -f "${run_out}/${srr}.fastq" ]] \
    || compgen -G "${run_out}/${srr}_*.fastq" >/dev/null 2>&1 \
    || compgen -G "${run_out}/${srr}.sra_*.fastq" >/dev/null 2>&1 \
    || [[ -f "${run_out}/${srr}.sra.fastq" ]]; then
    echo "[skip] ${srr}: FASTQ already in ${run_out}"
    continue
  fi

  echo "========== ${srr} =========="
  echo "prefetch -> ${run_out} (max size ${PREFETCH_MAX_SIZE})"
  prefetch "${srr}" -O "${run_out}" -X "${PREFETCH_MAX_SIZE}" -p

  sra_file="${run_out}/${srr}/${srr}.sra"
  if [[ ! -f "${sra_file}" ]]; then
    # Some prefetch layouts place .sra directly under run_out
    if compgen -G "${run_out}/${srr}.sra" >/dev/null; then
      sra_file="${run_out}/${srr}.sra"
    else
      sra_file="$(find "${run_out}" -name "${srr}.sra" -type f | head -1)"
    fi
  fi
  if [[ -z "${sra_file}" || ! -f "${sra_file}" ]]; then
    echo "Error: could not find ${srr}.sra under ${run_out}" >&2
    exit 1
  fi

  echo "fasterq-dump ${sra_file}"
  # Note: older sratoolkit (e.g. 2.10.x) has no --progress on fasterq-dump
  # -f: overwrite stale/partial outputs (otherwise rcExists / exit 3)
  fasterq-dump -f "${sra_file}" -O "${run_out}" -e "${THREADS}" --split-files

  # Remove .sra to save space (optional)
  rm -f "${sra_file}"
  find "${run_out}" -name "${srr}.sra" -delete 2>/dev/null || true
  # Remove empty prefetch subdir if left behind
  rmdir "${run_out}/${srr}" 2>/dev/null || true

  if [[ "${GZIP}" == "1" ]]; then
    for f in "${run_out}/${srr}"*.fastq; do
      [[ -f "$f" ]] || continue
      gzip -f "$f"
    done
  fi

  echo "[done] ${srr}"
  echo ""
done

echo "All finished. FASTQ under: ${OUT_DIR}"
