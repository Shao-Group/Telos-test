#!/bin/bash
ref_genome="/path/to/GRCh38.primary_assembly.genome.fa"
ref_annotation="/path/to/GRCh38_gencode.gtf"
gtfformat="/path/to/gtfformat"

cd $1

# Find all BAM files in the current directory
echo "Finding BAM files in directory: $(pwd)"
mapfile -t bam_files < <(find . -maxdepth 1 -name "*.bam" -type f | sort)

# Check if any BAM files were found
if [ ${#bam_files[@]} -eq 0 ]; then
    echo "Error: No BAM files found in directory: $(pwd)"
    exit 1
fi

# If only one BAM file, use it automatically
if [ ${#bam_files[@]} -eq 1 ]; then
    bam_file="${bam_files[0]}"
    echo "Found single BAM file: ${bam_file#./}"
    echo "Using: ${bam_file#./}"
else
    # Display available BAM files
    echo "Available BAM files:"
    for i in "${!bam_files[@]}"; do
        echo "$((i+1)). ${bam_files[i]#./}"
    done

    # Prompt user to select a BAM file
    while true; do
        read -p "Please select a BAM file (1-${#bam_files[@]}): " selection
        
        # Validate input
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#bam_files[@]} ]; then
            bam_file="${bam_files[$((selection-1))]}"
            echo "Selected: ${bam_file#./}"
            break
        else
            echo "Invalid selection. Please enter a number between 1 and ${#bam_files[@]}."
        fi
    done
fi

rm *refmap
rm *tmap
rm gffcmp/*
echo "Cleaned up old files"
echo "Current files:"
echo $(ls .)

echo "Generating stringtie_filtered.gtf..."
/path/to/stringtie -p 8 -L -o stringtie_filtered.gtf "$bam_file"
# echo "Generating stringtie_all.gtf..."
# /path/to/stringtie -p 8 -L -t -o stringtie_all.gtf "$bam_file"

eval "$(conda shell.bash hook)"

conda activate isoquant
isoquant_out="isoquant_out"
# echo "Generating isoquant_all.gtf..."

isoquant.py --threads 32 --model_construction_strategy all --reference $ref_genome --bam "$bam_file" --data_type pacbio -o $isoquant_out

cp "$isoquant_out/OUT/OUT.transcript_models.gtf" "isoquant_all.gtf"
cp "$isoquant_out/OUT/OUT.transcript_model_tpm.tsv" "isoquant_all.tpm"
rm -r $isoquant_out

echo "Generating isoquant_filtered.gtf..."
isoquant.py --threads 32 --reference $ref_genome --bam "$bam_file" --data_type pacbio -o $isoquant_out

cp "$isoquant_out/OUT/OUT.transcript_models.gtf" "isoquant_filtered.gtf"
cp "$isoquant_out/OUT/OUT.transcript_model_tpm.tsv" "isoquant_filtered.tpm"
rm -r $isoquant_out

conda deactivate

# "$gtfformat" update-tpm isoquant_all.gtf isoquant_all.tpm isoquant_all.gtf
"$gtfformat" update-tpm isoquant_filtered.gtf isoquant_filtered.tpm isoquant_filtered.gtf

conda activate gffcompare
mkdir -p gffcmp
cd gffcmp
echo "Running gffcompare for stringtie_filtered.gtf"
gffcompare -r $ref_annotation -o stringtie ../stringtie_filtered.gtf
cat stringtie.stats
echo "Running gffcompare for isoquant_filtered.gtf"
gffcompare -r $ref_annotation -o isoquant ../isoquant_filtered.gtf
cat isoquant.stats
conda deactivate

echo "completed gffcompare"





