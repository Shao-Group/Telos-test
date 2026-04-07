#!/usr/bin/env bash
# Build BED12 + splice-aware HISAT2 index for every annotation set under genome/{gencode,ensembl,refseq}/.
# Stops on first failure (set -euo pipefail). Logs each profile clearly.
#
# Usage:
#   ./scripts/prepare_all_genome_annotations.sh
#   THREADS=16 ./scripts/prepare_all_genome_annotations.sh
#   BED_ONLY=1 ./scripts/prepare_all_genome_annotations.sh   # BED only, no hisat2-build
#
# Not covered: genome/ucsc_knowgene (BED-style; no FASTA+GTF pair for this pipeline).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREADS="${THREADS:-8}"
EXTRA=( )
if [[ -n "${BED_ONLY:-}" && "${BED_ONLY}" != "0" ]]; then
  EXTRA=(--bed-only)
fi

for profile in gencode ensembl refseq; do
  echo ""
  echo "########################################################################"
  echo "# genome_prepare --profile ${profile} (threads=${THREADS})"
  echo "########################################################################"
  "${SCRIPT_DIR}/genome_prepare.sh" --profile "${profile}" --threads "${THREADS}" "${EXTRA[@]}"
done

echo ""
echo "All annotation profiles prepared (gencode, ensembl, refseq)."
