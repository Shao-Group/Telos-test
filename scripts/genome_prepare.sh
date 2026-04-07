#!/usr/bin/env bash
# Build RSeQC gene BED12 + splice-aware HISAT2 index from files in genome/.
#
# Prerequisites: python3, bedtools (sort), conda + env samtools-irtesam for samtools (faidx
#   if .fai missing), hisat2 2.x. Override: SAMTOOLS=/path/to/samtools or CONDA_SAMTOOLS_ENV=...
#
# Usage:
#   ./scripts/genome_prepare.sh --profile gencode [--threads 8]
#   ./scripts/genome_prepare.sh --profile ensembl [--threads 8]
#   ./scripts/genome_prepare.sh --profile refseq [--threads 8]
#   ./scripts/genome_prepare.sh --profile gencode --bed-only   # BED12 only (skip hisat2-build)
#
# Outputs (under genome/derived/<profile>/):
#   annotation.filtered.gtf   (gffread -E; optional -g when REF has matching contigs)
#   rseqc.transcripts.sorted.bed12
#   hisat2/<prefix>.*.ht2*    (HISAT2 index; prefix = grch38_primary_<profile>)

set -euo pipefail

GFFREAD="${GFFREAD:-/datadisk1/shared/tools/cufflinks-2.2.1/gffread}"
HISAT2_HOME="${HISAT2_HOME:-/datadisk1/shared/tools/hisat2/hisat2-2.2.0}"
BEDTOOLS="${BEDTOOLS:-/datadisk1/shared/tools/bedtools2/bin/bedtools}"
CONDA_SAMTOOLS_ENV="${CONDA_SAMTOOLS_ENV:-samtools-irtesam}"
# If SAMTOOLS is set, use that binary; else samtools from conda env CONDA_SAMTOOLS_ENV.
SAMTOOLS="${SAMTOOLS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENOME_DIR="${GENOME_DIR:-${REPO_ROOT}/genome}"
THREADS=8
PROFILE=""
BED_ONLY=0

usage() {
  sed -n '1,22p' "$0" | tail -n +2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --bed-only) BED_ONLY=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "${PROFILE}" ]] || { echo "Need --profile gencode|ensembl|refseq" >&2; exit 1; }

case "${PROFILE}" in
  gencode)
    REF_FASTA="${GENOME_DIR}/gencode/GRCh38.primary_assembly.genome.fa"
    REF_GTF="${GENOME_DIR}/gencode/gencode.v49.primary_assembly.basic.annotation.gtf"
    INDEX_PREFIX_NAME="grch38_primary_gencode49"
    ;;
  ensembl)
    REF_FASTA="${GENOME_DIR}/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    REF_GTF="${GENOME_DIR}/ensembl/Homo_sapiens.GRCh38.115.chr.gtf"
    INDEX_PREFIX_NAME="grch38_primary_ensembl115"
    ;;
  refseq)
    REF_FASTA="${GENOME_DIR}/refseq/GCF_000001405.40_GRCh38.p14_genomic.fna"
    REF_GTF="${GENOME_DIR}/refseq/GCF_000001405.40_GRCh38.p14_genomic.gtf"
    INDEX_PREFIX_NAME="grch38_p14_refseq"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 1
    ;;
esac

for f in "${REF_FASTA}" "${REF_GTF}"; do
  [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 1; }
done
[[ -x "${GFFREAD}" ]] || [[ -f "${GFFREAD}" ]] || { echo "gffread not found: ${GFFREAD}" >&2; exit 1; }
[[ -f "${HISAT2_HOME}/hisat2-build" ]] || { echo "hisat2-build not under HISAT2_HOME=${HISAT2_HOME}" >&2; exit 1; }
[[ -f "${HISAT2_HOME}/extract_splice_sites.py" ]] || { echo "Missing extract_splice_sites.py" >&2; exit 1; }
[[ -f "${HISAT2_HOME}/extract_exons.py" ]] || { echo "Missing extract_exons.py" >&2; exit 1; }

run_samtools() {
  if [[ -n "${SAMTOOLS}" ]]; then
    "${SAMTOOLS}" "$@"
  else
    conda run -n "${CONDA_SAMTOOLS_ENV}" --no-capture-output samtools "$@"
  fi
}

DERIVED="${GENOME_DIR}/derived/${PROFILE}"
INDEX_DIR="${DERIVED}/hisat2"
mkdir -p "${DERIVED}" "${INDEX_DIR}"

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

FAI="${REF_FASTA}.fai"
if [[ ! -f "${FAI}" ]]; then
  echo "Indexing FASTA -> ${FAI} (samtools via conda env ${CONDA_SAMTOOLS_ENV})"
  run_samtools faidx "${REF_FASTA}"
fi

FILTERED_GTF="${DERIVED}/annotation.filtered.gtf"
BED_RAW="${DERIVED}/rseqc.transcripts.bed12"
BED_SORTED="${DERIVED}/rseqc.transcripts.sorted.bed12"
SS_TXT="${DERIVED}/hisat2_splice_sites.txt"
EXON_TXT="${DERIVED}/hisat2_exons.txt"
INDEX_PREFIX="${INDEX_DIR}/${INDEX_PREFIX_NAME}"

echo "== gffread validate/filter -> ${FILTERED_GTF} (-T = GTF; required for transcript_id on exons)"
set +e
# -T: emit GTF (not default GFF3). GFF3 uses Parent= on exons and breaks gtf_to_bed12 + HISAT2 extract_*.py.
"${GFFREAD}" -E -T -g "${REF_FASTA}" "${REF_GTF}" -o "${FILTERED_GTF}" 2> "${DERIVED}/gffread.stderr"
GR=$?
set -e
if [[ ${GR} -ne 0 ]] || [[ ! -s "${FILTERED_GTF}" ]]; then
  echo "warn: gffread with -g failed (see ${DERIVED}/gffread.stderr); retry without -g"
  "${GFFREAD}" -E -T "${REF_GTF}" -o "${FILTERED_GTF}" 2>> "${DERIVED}/gffread.stderr"
  GR2=$?
  set -e
  if [[ ${GR2} -ne 0 ]] || [[ ! -s "${FILTERED_GTF}" ]]; then
    echo "ERROR: gffread failed for ${PROFILE}; see ${DERIVED}/gffread.stderr" >&2
    exit 1
  fi
fi

echo "== GTF -> BED12 -> sort (bedtools -g FAI)"
python3 "${SCRIPT_DIR}/gtf_to_bed12.py" "${FILTERED_GTF}" -o "${BED_RAW}"
[[ -s "${BED_RAW}" ]] || { echo "ERROR: ${BED_RAW} is empty after gtf_to_bed12.py" >&2; exit 1; }
"${BEDTOOLS}" sort -i "${BED_RAW}" -g "${FAI}" > "${BED_SORTED}"
[[ -s "${BED_SORTED}" ]] || { echo "ERROR: ${BED_SORTED} is empty after bedtools sort" >&2; exit 1; }

if [[ "${BED_ONLY}" -eq 1 ]]; then
  echo "== (--bed-only) skipping HISAT2 index"
  echo "Done."
  echo "  RSeQC BED12:     ${BED_SORTED}"
  echo "  Filtered GTF:    ${FILTERED_GTF}"
  exit 0
fi

echo "== HISAT2 extract splice sites / exons"
python3 "${HISAT2_HOME}/extract_splice_sites.py" "${FILTERED_GTF}" > "${SS_TXT}"
python3 "${HISAT2_HOME}/extract_exons.py" "${FILTERED_GTF}" > "${EXON_TXT}"
[[ -s "${SS_TXT}" ]] || { echo "ERROR: splice sites file empty: ${SS_TXT}" >&2; exit 1; }
[[ -s "${EXON_TXT}" ]] || { echo "ERROR: exons file empty: ${EXON_TXT}" >&2; exit 1; }

echo "== hisat2-build (splice-aware)"
"${HISAT2_HOME}/hisat2-build" -p "${THREADS}" \
  --ss "${SS_TXT}" --exon "${EXON_TXT}" \
  "${REF_FASTA}" "${INDEX_PREFIX}"

echo "Done."
echo "  RSeQC BED12:     ${BED_SORTED}"
echo "  HISAT2 index -x: ${INDEX_PREFIX}"
echo "  Filtered GTF:    ${FILTERED_GTF}"
