#!/usr/bin/env bash
# Primary entry: run rnaseq_pipeline over fastq/ and emit Telos data bundles (BAM, GTFs, gffcmp, bundle_manifest.yaml).
# Strand inference is not part of this script; short reads use HISAT2_STRANDNESS (default RF).
#
# fastq/ layout (this repo):
#   sr/SRR*/           paired short reads → --preset short_paired + HISAT2
#   cdna/*.fastq       ONT cDNA          → --preset nanopore --nanopore-library cdna
#   drna/*.fastq       ONT direct RNA    → --preset nanopore --nanopore-library drna
#   pacbio/*.fastq     PacBio Iso-Seq    → --preset pacbio
#
# Output layout (default):
#   data/bundles/${REF_ID}/sr/<SRR>/
#   data/bundles/${REF_ID}/ont_cdna/<sample>/
#   data/bundles/${REF_ID}/ont_drna/<sample>/
#   data/bundles/${REF_ID}/pacbio/<sample>/
#
# Each work-dir gets: align/, stringtie.gtf, scallop2.gtf|isoquant.gtf, gffcmp/, bundle_manifest.yaml
#
# Reference sets live under genome/{gencode,ensembl,refseq}/ (see scripts/genome_prepare.sh).
# ANNOTATION_PROFILE picks FASTA+GTF+HISAT2 index + bundle root, or "all" runs three passes.
#
# Usage:
#   ./scripts/run_all_fastq_bundles.sh
#   ANNOTATION_PROFILE=ensembl ./scripts/run_all_fastq_bundles.sh
#   ANNOTATION_PROFILE=all ./scripts/run_all_fastq_bundles.sh
#   DRY_RUN=1 ./scripts/run_all_fastq_bundles.sh
#   SKIP_EXISTING=1 ./scripts/run_all_fastq_bundles.sh
#   ONLY=sr ./scripts/run_all_fastq_bundles.sh          # sr | cdna | drna | pacbio
#   HISAT2_STRANDNESS=RF THREADS=16 ./scripts/run_all_fastq_bundles.sh
#
# Requires: conda env CONDA_ENV; run scripts/prepare_all_genome_annotations.sh (or genome_prepare per profile).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"

CONDA_ENV="${CONDA_ENV:-irtesam-berth}"
FASTQ_ROOT="${FASTQ_ROOT:-${REPO_ROOT}/fastq}"

THREADS="${THREADS:-8}"
THREADS_ISO="${THREADS_ISO:-32}"
ONLY="${ONLY:-all}"
ANNOTATION_PROFILE="${ANNOTATION_PROFILE:-gencode}"
# Default RF when unset; use HISAT2_STRANDNESS= to align short reads unstranded (- only, not :-).
HISAT2_STRANDNESS="${HISAT2_STRANDNESS-RF}"

if [[ "${ANNOTATION_PROFILE}" == all ]]; then
  ec=0
  for ap in gencode ensembl refseq; do
    echo ""
    echo "#####################################################################"
    echo "# ANNOTATION_PROFILE=${ap} (sub-run)"
    echo "#####################################################################"
    ANNOTATION_PROFILE="${ap}" "${BASH_SOURCE[0]}" "$@" || ec=1
  done
  exit "${ec}"
fi

case "${ANNOTATION_PROFILE}" in
  gencode|ensembl|refseq) ;;
  *)
    echo "ERROR: ANNOTATION_PROFILE must be gencode|ensembl|refseq|all, got: ${ANNOTATION_PROFILE}" >&2
    exit 1
    ;;
esac

apply_annotation_profile() {
  case "$1" in
    gencode)
      REF_ID="GRCh38_gencode49"
      REF_FASTA="${REPO_ROOT}/genome/gencode/GRCh38.primary_assembly.genome.fa"
      REF_GTF="${REPO_ROOT}/genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf"
      HISAT2_INDEX="${REPO_ROOT}/genome/derived/gencode/hisat2/grch38_primary_gencode49"
      ;;
    ensembl)
      REF_ID="GRCh38_ensembl115"
      REF_FASTA="${REPO_ROOT}/genome/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
      REF_GTF="${REPO_ROOT}/genome/ensembl/Homo_sapiens.GRCh38.115.chr.gtf"
      HISAT2_INDEX="${REPO_ROOT}/genome/derived/ensembl/hisat2/grch38_primary_ensembl115"
      ;;
    refseq)
      REF_ID="GRCh38_refseq_p14"
      REF_FASTA="${REPO_ROOT}/genome/refseq/GCF_000001405.40_GRCh38.p14_genomic.fna"
      REF_GTF="${REPO_ROOT}/genome/refseq/GCF_000001405.40_GRCh38.p14_genomic.gtf"
      HISAT2_INDEX="${REPO_ROOT}/genome/derived/refseq/hisat2/grch38_p14_refseq"
      ;;
  esac
  BUNDLE_ROOT="${REPO_ROOT}/data/bundles/${REF_ID}"
}

apply_annotation_profile "${ANNOTATION_PROFILE}"

MANIFEST_REF_GTF="${REF_GTF}"
PIPELINE_REF_GTF="${REF_GTF}"

ensure_refseq_gffcompare_gtf() {
  # RefSeq GTF can contain records that make gffcompare abort ("no valid ID found").
  # Create a gffcompare-safe reference GTF containing only transcript/exon lines
  # with both gene_id and transcript_id.
  local in_gtf="$1"
  local out_gtf="$2"
  if [[ -f "${out_gtf}" ]] && [[ "${out_gtf}" -nt "${in_gtf}" ]]; then
    return 0
  fi
  echo "[refseq] building gffcompare-safe GTF: ${out_gtf}"
  awk 'BEGIN{FS=OFS="\t"}
       /^#/ {print; next}
       $3!="exon" && $3!="transcript" {next}
       $9 ~ /gene_id/ && $9 ~ /transcript_id/ {print}' \
    "${in_gtf}" > "${out_gtf}"
  if [[ ! -s "${out_gtf}" ]]; then
    echo "ERROR: filtered RefSeq GTF is empty: ${out_gtf}" >&2
    exit 1
  fi
}

if [[ "${ANNOTATION_PROFILE}" == refseq ]]; then
  REFSEQ_GFFCOMPARE_GTF="${REPO_ROOT}/genome/refseq/GCF_000001405.40_GRCh38.p14_genomic.gffcmp.gtf"
  ensure_refseq_gffcompare_gtf "${REF_GTF}" "${REFSEQ_GFFCOMPARE_GTF}"
  PIPELINE_REF_GTF="${REFSEQ_GFFCOMPARE_GTF}"
fi

is_complete_workdir() {
  # Define "complete" as having the expected outputs *and* a manifest.
  local work_dir="$1"
  local preset="$2"
  [[ -f "${work_dir}/bundle_manifest.yaml" ]] || return 1
  [[ -f "${work_dir}/align/aln.sorted.bam" ]] || return 1
  [[ -f "${work_dir}/stringtie.gtf" ]] || return 1
  [[ -f "${work_dir}/gffcmp/stringtie.stats" ]] || return 1
  case "${preset}" in
    short_paired)
      [[ -f "${work_dir}/scallop2.gtf" ]] || return 1
      [[ -f "${work_dir}/gffcmp/scallop2.stats" ]] || return 1
      ;;
    nanopore|pacbio)
      [[ -f "${work_dir}/isoquant.gtf" ]] || return 1
      [[ -f "${work_dir}/gffcmp/isoquant.stats" ]] || return 1
      ;;
  esac
  return 0
}

run_pipeline() {
  local work_dir="$1"
  shift
  local preset="${1:-}"
  shift || true
  mkdir -p "${work_dir}"
  # If a previous run died mid-way, keep BAM/assembly outputs but wipe partial gffcompare outputs.
  if [[ -d "${work_dir}/gffcmp" ]] && [[ ! -f "${work_dir}/bundle_manifest.yaml" ]]; then
    rm -rf "${work_dir}/gffcmp"
  fi
  if [[ -n "${SKIP_EXISTING:-}" ]] && is_complete_workdir "${work_dir}" "${preset}"; then
    echo "[skip] complete bundle ${work_dir}"
    return 0
  fi
  local -a cmd=( -m rnaseq_pipeline run --work-dir "${work_dir}" "$@" )
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN: conda run -n ${CONDA_ENV} --no-capture-output python ${cmd[*]}"
    return 0
  fi
  conda run -n "${CONDA_ENV}" --no-capture-output python "${cmd[@]}"
}

# Always run after a sample finishes (or SKIP_EXISTING left a complete tree). Fails loudly on error.
require_manifest() {
  local work_dir="$1" sample_id="$2" aligner="$3"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "[dry-run] write_bundle_manifest ${work_dir}"
    return 0
  fi
  if [[ ! -f "${work_dir}/align/aln.sorted.bam" ]]; then
    echo "ERROR: require_manifest: missing ${work_dir}/align/aln.sorted.bam (sample ${sample_id})" >&2
    exit 1
  fi
  python3 "${SCRIPT_DIR}/write_bundle_manifest.py" \
    --work-dir "${work_dir}" \
    --ref-id "${REF_ID}" \
    --sample-id "${sample_id}" \
    --genome-fasta "${REF_FASTA}" \
    --ref-gtf "${MANIFEST_REF_GTF}" \
    --aligner "${aligner}" \
    || {
      echo "ERROR: write_bundle_manifest.py failed for ${work_dir} (sample ${sample_id})" >&2
      exit 1
    }
}

check_short_prereqs() {
  [[ -f "${REF_FASTA}" ]] || { echo "Missing REF_FASTA=${REF_FASTA}" >&2; exit 1; }
  [[ -f "${REF_GTF}" ]] || { echo "Missing REF_GTF=${REF_GTF}" >&2; exit 1; }
  if ! compgen -G "${HISAT2_INDEX}."*.ht2l > /dev/null 2>&1 && ! compgen -G "${HISAT2_INDEX}."*.ht2 > /dev/null 2>&1; then
    echo "ERROR: Missing HISAT2 index at ${HISAT2_INDEX}.*.ht2(l); run scripts/genome_prepare.sh --profile ${ANNOTATION_PROFILE}" >&2
    exit 1
  fi
}

check_long_prereqs() {
  [[ -f "${REF_FASTA}" ]] || { echo "Missing REF_FASTA=${REF_FASTA}" >&2; exit 1; }
  [[ -f "${REF_GTF}" ]] || { echo "Missing REF_GTF=${REF_GTF}" >&2; exit 1; }
}

should_run() {
  local section="$1"
  [[ "${ONLY}" == all || "${ONLY}" == "${section}" ]]
}

echo "REPO_ROOT=${REPO_ROOT}"
echo "ANNOTATION_PROFILE=${ANNOTATION_PROFILE}"
echo "REF_ID=${REF_ID}"
echo "REF_FASTA=${REF_FASTA}"
echo "REF_GTF=${REF_GTF}"
echo "PIPELINE_REF_GTF=${PIPELINE_REF_GTF}"
echo "HISAT2_INDEX=${HISAT2_INDEX}"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "ONLY=${ONLY} SKIP_EXISTING=${SKIP_EXISTING:-} DRY_RUN=${DRY_RUN:-}"
echo ""

# --- Short reads (paired dirs under fastq/sr/SRR*) ---
if should_run sr; then
  check_short_prereqs
  shopt -s nullglob
  for sdir in "${FASTQ_ROOT}/sr"/SRR[0-9]*; do
    [[ -d "${sdir}" ]] || continue
    srr="$(basename "${sdir}")"
    work="${BUNDLE_ROOT}/sr/${srr}"
    echo "========== ${srr} (short_paired) -> ${work}"
    hs_args=( )
    if [[ -n "${HISAT2_STRANDNESS}" ]]; then
      hs_args=( --hisat2-strandness "${HISAT2_STRANDNESS}" )
    fi
    run_pipeline "${work}" short_paired \
      --preset short_paired \
      --fastq "${sdir}" \
      --ref-fasta "${REF_FASTA}" \
      --ref-gtf "${PIPELINE_REF_GTF}" \
      --hisat2-index "${HISAT2_INDEX}" \
      "${hs_args[@]}" \
      --threads-align "${THREADS}" \
      --threads-assembly "${THREADS}" \
      --threads-isoquant "${THREADS_ISO}"
    require_manifest "${work}" "${srr}" hisat2
    echo ""
  done
  shopt -u nullglob
fi

# --- ONT cDNA ---
if should_run cdna; then
  check_long_prereqs
  shopt -s nullglob
  for fq in "${FASTQ_ROOT}/cdna"/*.fastq "${FASTQ_ROOT}/cdna"/*.fq "${FASTQ_ROOT}/cdna"/*.fastq.gz "${FASTQ_ROOT}/cdna"/*.fq.gz; do
    [[ -f "${fq}" ]] || continue
    sid="$(basename "${fq}")"
    sid="${sid%.fastq.gz}"
    sid="${sid%.fq.gz}"
    sid="${sid%.fastq}"
    sid="${sid%.fq}"
    work="${BUNDLE_ROOT}/ont_cdna/${sid}"
    echo "========== ${sid} (nanopore cdna) -> ${work}"
    run_pipeline "${work}" nanopore \
      --preset nanopore \
      --nanopore-library cdna \
      --fastq "${fq}" \
      --ref-fasta "${REF_FASTA}" \
      --ref-gtf "${PIPELINE_REF_GTF}" \
      --threads-align "${THREADS}" \
      --threads-assembly "${THREADS}" \
      --threads-isoquant "${THREADS_ISO}"
    require_manifest "${work}" "${sid}" minimap2
    echo ""
  done
  shopt -u nullglob
fi

# --- ONT dRNA ---
if should_run drna; then
  check_long_prereqs
  shopt -s nullglob
  for fq in "${FASTQ_ROOT}/drna"/*.fastq "${FASTQ_ROOT}/drna"/*.fq "${FASTQ_ROOT}/drna"/*.fastq.gz "${FASTQ_ROOT}/drna"/*.fq.gz; do
    [[ -f "${fq}" ]] || continue
    sid="$(basename "${fq}")"
    sid="${sid%.fastq.gz}"
    sid="${sid%.fq.gz}"
    sid="${sid%.fastq}"
    sid="${sid%.fq}"
    work="${BUNDLE_ROOT}/ont_drna/${sid}"
    echo "========== ${sid} (nanopore drna) -> ${work}"
    run_pipeline "${work}" nanopore \
      --preset nanopore \
      --nanopore-library drna \
      --fastq "${fq}" \
      --ref-fasta "${REF_FASTA}" \
      --ref-gtf "${PIPELINE_REF_GTF}" \
      --threads-align "${THREADS}" \
      --threads-assembly "${THREADS}" \
      --threads-isoquant "${THREADS_ISO}"
    require_manifest "${work}" "${sid}" minimap2
    echo ""
  done
  shopt -u nullglob
fi

# --- PacBio ---
if should_run pacbio; then
  check_long_prereqs
  shopt -s nullglob
  for fq in "${FASTQ_ROOT}/pacbio"/*.fastq "${FASTQ_ROOT}/pacbio"/*.fq "${FASTQ_ROOT}/pacbio"/*.fastq.gz "${FASTQ_ROOT}/pacbio"/*.fq.gz; do
    [[ -f "${fq}" ]] || continue
    sid="$(basename "${fq}")"
    sid="${sid%.fastq.gz}"
    sid="${sid%.fq.gz}"
    sid="${sid%.fastq}"
    sid="${sid%.fq}"
    work="${BUNDLE_ROOT}/pacbio/${sid}"
    echo "========== ${sid} (pacbio) -> ${work}"
    run_pipeline "${work}" pacbio \
      --preset pacbio \
      --fastq "${fq}" \
      --ref-fasta "${REF_FASTA}" \
      --ref-gtf "${PIPELINE_REF_GTF}" \
      --threads-align "${THREADS}" \
      --threads-assembly "${THREADS}" \
      --threads-isoquant "${THREADS_ISO}"
    require_manifest "${work}" "${sid}" minimap2
    echo ""
  done
  shopt -u nullglob
fi

echo "All requested sections finished. Bundles under: ${BUNDLE_ROOT}"
echo "Next: point Telos/install at each bundle_manifest.yaml (or merge into experiment YAML)."
