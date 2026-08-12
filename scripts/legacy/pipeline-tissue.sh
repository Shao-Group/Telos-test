export REPO=/datadisk1/ixk5174/project_repo/Telos-test
cd "$REPO"

export REFA="$REPO/genome/gencode/GRCh38.primary_assembly.genome.fa"
export REFG="$REPO/genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf"
export HS2="$REPO/genome/derived/gencode/hisat2/grch38_primary_gencode49"
export THREADS=20

# 1) ONT cDNA tissue → data/tissue/cdna
PYTHONPATH=src python3 -m rnaseq_pipeline run \
  --preset nanopore \
  --fastq "$REPO/fastq/tissue/cdna" \
  --work-dir "$REPO/data/tissue/cdna" \
  --ref-fasta "$REFA" \
  --ref-gtf "$REFG" \
  --conda-env irtesam-berth \
  --isoquant-conda-env isoquant \
  --isoquant-script isoquant.py \
  --threads-align $THREADS  \
  --threads-assembly $THREADS \
  --threads-isoquant $THREADS

# 2) ONT direct RNA tissue → data/tissue/drna
PYTHONPATH=src python3 -m rnaseq_pipeline run \
  --preset nanopore \
  --nanopore-library drna \
  --fastq "$REPO/fastq/tissue/drna" \
  --work-dir "$REPO/data/tissue/drna" \
  --ref-fasta "$REFA" \
  --ref-gtf "$REFG" \
  --conda-env irtesam-berth \
  --isoquant-conda-env isoquant \
  --isoquant-script isoquant.py \
  --threads-align $THREADS \
  --threads-assembly $THREADS \
  --threads-isoquant $THREADS

# 3) PacBio tissue → data/tissue/pacbio
PYTHONPATH=src python3 -m rnaseq_pipeline run \
  --preset pacbio \
  --fastq "$REPO/fastq/tissue/pacbio" \
  --work-dir "$REPO/data/tissue/pacbio" \
  --ref-fasta "$REFA" \
  --ref-gtf "$REFG" \
  --conda-env irtesam-berth \
  --isoquant-conda-env isoquant \
  --isoquant-script isoquant.py \
  --threads-align $THREADS \
  --threads-assembly $THREADS \
  --threads-isoquant $THREADS

# 4) Short paired-end tissue → data/tissue/sr
PYTHONPATH=src python3 -m rnaseq_pipeline run \
  --preset short_paired \
  --fastq "$REPO/fastq/tissue/sr" \
  --work-dir "$REPO/data/tissue/sr" \
  --ref-fasta "$REFA" \
  --ref-gtf "$REFG" \
  --hisat2-index "$HS2" \
  --hisat2-strandness RF \
  --conda-env irtesam-berth \
  --threads-align $THREADS \
  --threads-assembly $THREADS