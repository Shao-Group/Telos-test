#!/bin/bash
ref_genome="/datadisk1/ixk5174/data/long_reads_assembly/human_ref.Gencode.v36/GRCh38.primary_assembly.genome.fa"
ref_annotation="/datadisk1/ixk5174/project_repo/Telos-test/data/GRCh38_ensembl.gtf"
gtfformat="/datadisk1/ixk5174/tools/rnaseqtools/gtfformat/gtfformat"

# Source and destination directories
encode10_dir="/datadisk1/shared/data/encode10"
data_dir="/datadisk1/ixk5174/project_repo/Telos-test/data"

# Find all SRR directories in ENCODE10
echo "Finding SRR directories in: $encode10_dir"
mapfile -t srr_dirs < <(find "$encode10_dir" -maxdepth 1 -type d -name "SRR[0-9]*" | sort)

# Check if any SRR directories were found
if [ ${#srr_dirs[@]} -eq 0 ]; then
    echo "Error: No SRR directories found in: $encode10_dir"
    exit 1
fi

echo "Found ${#srr_dirs[@]} SRR directories:"
for srr_dir in "${srr_dirs[@]}"; do
    echo "  $(basename "$srr_dir")"
done

# Process each SRR directory
for srr_dir in "${srr_dirs[@]}"; do
    srr_name=$(basename "$srr_dir")
    target_dir="$data_dir/cv_${srr_name}_hisat"
    
    echo ""
    echo "=========================================="
    echo "Processing: $srr_name"
    echo "Source: $srr_dir"
    echo "Target: $target_dir"
    echo "=========================================="
    
    # Check if target directory already exists
    if [ -d "$target_dir" ]; then
        echo "Target directory already exists: $target_dir"
        echo "Skipping $srr_name..."
        continue
    fi
    
    echo "$(ls $srr_dir)"
    # Find BAM and BAI files in source directory
    echo "Looking for BAM and BAI files in: $srr_dir"
    bam_files=($(find "$srr_dir" -name "hisat.sort.bam"))
    bai_files=($(find "$srr_dir" -name "hisat.sort.bam.bai" ))
    
    if [ ${#bam_files[@]} -eq 0 ]; then
        echo "Warning: No BAM files found in $srr_dir"
        continue
    fi


    # Create target directory
    echo "Creating target directory: $target_dir"
    mkdir -p "$target_dir"
    
    # Copy BAM files
    echo "Copying BAM files..."
    for bam_file in "${bam_files[@]}"; do
        echo "  Copying: $(basename "$bam_file")"
        cp "$bam_file" "$target_dir/"
    done
    
    # Copy BAI files
    echo "Copying BAI files..."
    for bai_file in "${bai_files[@]}"; do
        echo "  Copying: $(basename "$bai_file")"
        cp "$bai_file" "$target_dir/"
    done
    
    # Change to target directory
    cd "$target_dir"
    
    # Find the BAM file to use (should be hisat.sort.bam)
    bam_file=""
    
    if [ -f "hisat.sort.bam" ]; then
        bam_file="hisat.sort.bam"
    else
        # Use the first BAM file found as fallback
        bam_file=$(find . -maxdepth 1 -name "*.bam" -type f | head -1)
        bam_file=$(basename "$bam_file")
    fi
    
    if [ -z "$bam_file" ] || [ ! -f "$bam_file" ]; then
        echo "Error: No suitable BAM file found in $target_dir"
        continue
    fi
    
    echo "Using BAM file: $bam_file"
    
    # Clean up old files
    rm -f *refmap *tmap
    rm -rf gffcmp
    echo "Cleaned up old files"
    
    # Run StringTie
    echo "Generating stringtie_filtered.gtf..."
    /datadisk1/ixk5174/tools/stringtie-3.0.0.Linux_x86_64/stringtie -p 8 -o stringtie_filtered.gtf "$bam_file"
    
    # Activate conda environment for scallop2
    eval "$(conda shell.bash hook)"
    conda activate scallop2
    
    # Run Scallop2
    echo "Generating scallop2_filtered.gtf..."
    scallop2 -i "$bam_file" -o scallop2_filtered.gtf
    
    conda deactivate
    
    # Activate conda environment for gffcompare
    conda activate gffcompare
    
    # Run gffcompare
    mkdir -p gffcmp
    cd gffcmp
    echo "Running gffcompare for stringtie_filtered.gtf"
    gffcompare -r "$ref_annotation" -o stringtie ../stringtie_filtered.gtf
    cat stringtie.stats
    echo "Running gffcompare for scallop2_filtered.gtf"
    gffcompare -r "$ref_annotation" -o scallop2 ../scallop2_filtered.gtf
    cat scallop2.stats
    echo "Completed gffcompare for $srr_name"
    
    conda deactivate
    
    # Return to original directory
    cd "$data_dir"
    
    echo "Completed processing: $srr_name"
done

echo ""
echo "=========================================="
echo "All SRR directories processed successfully!"
echo "=========================================="






