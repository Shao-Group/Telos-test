#!/usr/bin/env bash
# Subsample paired FASTQ (no seqtk: use head), align unstranded with HISAT2, run RSeQC infer_experiment.
#
# Requires: hisat2, conda env samtools-irtesam for samtools, infer_experiment.py on PATH
#
# Usage:
#   ./scripts/infer_strandedness_sr_fastq.sh \
#     --profile gencode \
#     --fastq-dir /path/to/fastq/sr/SRR307903 \
#     --out-dir /path/to/work/infer_SRR307903
#
# Genome bundle must already exist (run scripts/genome_prepare.sh for the same --profile).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENOME_DIR="${GENOME_DIR:-${REPO_ROOT}/genome}"

HISAT2_HOME="${HISAT2_HOME:-/datadisk1/shared/tools/hisat2/hisat2-2.2.0}"
CONDA_SAMTOOLS_ENV="${CONDA_SAMTOOLS_ENV:-samtools-irtesam}"
SAMTOOLS="${SAMTOOLS:-}"

run_samtools() {
  if [[ -n "${SAMTOOLS}" ]]; then
    "${SAMTOOLS}" "$@"
  else
    conda run -n "${CONDA_SAMTOOLS_ENV}" --no-capture-output samtools "$@"
  fi
}

if [[ -n "${SAMTOOLS}" ]]; then
  if [[ ! -x "${SAMTOOLS}" ]] && ! command -v "${SAMTOOLS}" >/dev/null 2>&1; then
    echo "SAMTOOLS is not executable: ${SAMTOOLS}" >&2
    exit 1
  fi
else
  command -v conda >/dev/null 2>&1 || { echo "conda not on PATH; set SAMTOOLS=/path/to/samtools" >&2; exit 1; }
  conda run -n "${CONDA_SAMTOOLS_ENV}" --no-capture-output samtools --version >/dev/null 2>&1 \
    || { echo "Cannot run samtools in conda env ${CONDA_SAMTOOLS_ENV}" >&2; exit 1; }
fi
# Lines to keep per mate (4 lines per read). Default 400000 = 100000 read pairs.
SUBSAMPLE_LINES="${SUBSAMPLE_LINES:-400000}"

PROFILE=""
FASTQ_DIR=""
OUT_DIR=""
THREADS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --fastq-dir) FASTQ_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${PROFILE}" && -n "${FASTQ_DIR}" && -n "${OUT_DIR}" ]] || {
  echo "Usage: $0 --profile gencode|ensembl|refseq --fastq-dir DIR --out-dir DIR [--threads N]" >&2
  exit 1
}

DERIVED="${GENOME_DIR}/derived/${PROFILE}"
BED="${DERIVED}/rseqc.transcripts.sorted.bed12"
case "${PROFILE}" in
  gencode) REF_FASTA="${GENOME_DIR}/gencode/GRCh38.primary_assembly.genome.fa"; INDEX_NAME="grch38_primary_gencode49" ;;
  ensembl) REF_FASTA="${GENOME_DIR}/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"; INDEX_NAME="grch38_primary_ensembl115" ;;
  refseq) REF_FASTA="${GENOME_DIR}/refseq/GCF_000001405.40_GRCh38.p14_genomic.fna"; INDEX_NAME="grch38_p14_refseq" ;;
  *) echo "Bad profile" >&2; exit 1 ;;
esac

INDEX_PREFIX="${DERIVED}/hisat2/${INDEX_NAME}"

[[ -f "${BED}" ]] || { echo "Missing ${BED}; run scripts/genome_prepare.sh --profile ${PROFILE}" >&2; exit 1; }
[[ -f "${REF_FASTA}" ]] || { echo "Missing REF ${REF_FASTA}" >&2; exit 1; }
if ! compgen -G "${INDEX_PREFIX}."*.ht2l > /dev/null 2>&1 && ! compgen -G "${INDEX_PREFIX}."*.ht2 > /dev/null 2>&1; then
  echo "Missing HISAT2 index beside ${INDEX_PREFIX} (expected .ht2l or .ht2). Run scripts/genome_prepare.sh --profile ${PROFILE}" >&2
  exit 1
fi

FASTQ_DIR="$(cd "${FASTQ_DIR}" && pwd)"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

shopt -s nullglob
r1=( "${FASTQ_DIR}"/*_1.fastq "${FASTQ_DIR}"/*.R1.fastq "${FASTQ_DIR}"/*.r1.fastq )
r2=( "${FASTQ_DIR}"/*_2.fastq "${FASTQ_DIR}"/*.R2.fastq "${FASTQ_DIR}"/*.r2.fastq )
shopt -u nullglob

pick1="${r1[0]:-}"
pick2="${r2[0]:-}"
[[ -n "${pick1}" && -n "${pick2}" ]] || {
  echo "Could not find *_1.fastq and *_2.fastq under ${FASTQ_DIR}" >&2
  exit 1
}
if [[ ${#r1[@]} -gt 1 || ${#r2[@]} -gt 1 ]]; then
  echo "warn: multiple R1/R2 files; using first pair only: ${pick1} + ${pick2}" >&2
fi

SUB_R1="${OUT_DIR}/sub_1.fastq"
SUB_R2="${OUT_DIR}/sub_2.fastq"
BAM="${OUT_DIR}/infer.sort.bam"
LOG="${OUT_DIR}/infer_experiment.txt"

# HISAT2 wrapper execs hisat2_read_statistics.py; kernel runs: python2 <that.py> <read1.fastq>.
# The bundled script is Python 2-only; running it under python3 causes SyntaxError.
# Prefer real python2; else PATH shim runs our Py3 port with the FASTQ path ($2).
PYTHON2_SHIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/infer_strand_py2shim.XXXXXX")"
cleanup_py2_shim() { rm -rf "${PYTHON2_SHIM_DIR}"; }
trap cleanup_py2_shim EXIT
PY3_STATS="${SCRIPT_DIR}/hisat2_read_statistics_py3.py"
if command -v python2.7 >/dev/null 2>&1; then
  printf '%s\n' '#!/bin/sh' 'exec python2.7 "$@"' > "${PYTHON2_SHIM_DIR}/python2"
elif command -v python2 >/dev/null 2>&1; then
  printf '%s\n' '#!/bin/sh' 'exec python2 "$@"' > "${PYTHON2_SHIM_DIR}/python2"
elif [[ -f "${PY3_STATS}" ]]; then
  printf '%s\n' '#!/bin/sh' "exec python3 '${PY3_STATS}' \"\$2\"" > "${PYTHON2_SHIM_DIR}/python2"
else
  printf '%s\n' '#!/bin/sh' 'exec python3 "$@"' > "${PYTHON2_SHIM_DIR}/python2"
  echo "warn: no python2 and missing ${PY3_STATS}; read-length check may error" >&2
fi
chmod +x "${PYTHON2_SHIM_DIR}/python2"
export PATH="${PYTHON2_SHIM_DIR}:${PATH}"

echo "Subsampling (first ${SUBSAMPLE_LINES} lines per mate) ..."
head -n "${SUBSAMPLE_LINES}" "${pick1}" > "${SUB_R1}"
head -n "${SUBSAMPLE_LINES}" "${pick2}" > "${SUB_R2}"

echo "HISAT2 (unstranded, --dta) -> sorted BAM (samtools via conda env ${CONDA_SAMTOOLS_ENV})"
"${HISAT2_HOME}/hisat2" -p "${THREADS}" -x "${INDEX_PREFIX}" --dta \
  -1 "${SUB_R1}" -2 "${SUB_R2}" \
  | run_samtools sort -@ "${THREADS}" -o "${BAM}"
run_samtools index "${BAM}"

echo "RSeQC infer_experiment.py -> ${LOG}"
infer_experiment.py -i "${BAM}" -r "${BED}" | tee "${LOG}"

echo "Done. Interpret ${LOG} (fr-firststrand -> usually HISAT2 RF; fr-secondstrand -> FR)."
