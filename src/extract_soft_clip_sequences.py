#!/usr/bin/env python3
"""
Extract soft-clipped sequences from BAM files for CNN training.
This script extracts individual left and right soft-clipped sequences from candidate sites.
"""

import argparse
import pysam
import pandas as pd
import numpy as np
import os
import sys
from config import load_config, save_config
from tqdm import tqdm
import gen_baseline_labels as gen_baseline_labels


def load_candidate_sites(file_path):
    """Load candidate sites from file."""
    with open(file_path, 'r') as f:
        candidate_sites_text = f.read()
    
    tss_candidate_sites = []
    tes_candidate_sites = []
    
    for line in candidate_sites_text.strip().split("\n"):
        parts = line.strip().split()
        site_type, chrom, pos, pos_strand_count, neg_strand_count = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4])
        strand = "+" if pos_strand_count > neg_strand_count else "-" 
        if site_type == "TSS":
            tss_candidate_sites.append((chrom, pos, strand))
        elif site_type == "TES":
            tes_candidate_sites.append((chrom, pos, strand))

    return tss_candidate_sites, tes_candidate_sites


def compute_strand(read):
    """Compute read strand considering ts tag."""
    try:
        ts = read.get_tag("ts")
        if ts not in ["+", "-"]:
            return "."
        if read.is_reverse:
            return "-" if ts == "+" else "+"
        else:
            return "+" if ts == "+" else "-"
    except KeyError:
        # No ts tag present, infer from read orientation
        return "-" if read.is_reverse else "+"


def extract_soft_clip_sequences(bam, chrom, pos, strand, cfg):
    """
    Extract individual soft-clipped sequences from reads around a candidate site.
    
    Args:
        bam: pysam.AlignmentFile object
        chrom: chromosome name
        pos: position of candidate site
        strand: strand of candidate site
        cfg: configuration object
        
    Returns:
        list of dictionaries containing sequence information
    """
    region_start = max(0, pos - cfg.window_size)
    region_end = pos + cfg.window_size
    reads = bam.fetch(chrom, region_start, region_end)

    sequences = []
    
    for read in reads:
        if read.is_unmapped or read.mapping_quality < cfg.min_mapq:
            continue
            
        read_strand = compute_strand(read)
        
        # Extract soft clip information
        if read.cigartuples and read.query_sequence:
            left_soft_clip_length = 0
            right_soft_clip_length = 0
            left_soft_clip_seq = ""
            right_soft_clip_seq = ""
            
            # Check for soft clips
            if read.cigartuples[0][0] == 4:  # Soft clip at start
                left_soft_clip_length = read.cigartuples[0][1]
                left_soft_clip_seq = read.query_sequence[:left_soft_clip_length]
            
            if read.cigartuples[-1][0] == 4:  # Soft clip at end
                right_soft_clip_length = read.cigartuples[-1][1]
                right_soft_clip_seq = read.query_sequence[-right_soft_clip_length:]
            
            # Process left soft clip
            if left_soft_clip_seq and len(left_soft_clip_seq) >= 2:  # Minimum length filter
                seq_info = {
                    'chrom': chrom,
                    'position': pos,
                    'strand': strand,
                    'sequence': left_soft_clip_seq,
                    'clip_type': 'start' if read_strand == '+' else 'end',
                    'read_strand': read_strand,
                    'read_position': read.reference_start,
                    'distance_to_site': abs(read.reference_start - pos),
                    'sequence_length': len(left_soft_clip_seq),
                    'mapping_quality': read.mapping_quality
                }
                sequences.append(seq_info)
            
            # Process right soft clip
            if right_soft_clip_seq and len(right_soft_clip_seq) >= 2:  # Minimum length filter
                seq_info = {
                    'chrom': chrom,
                    'position': pos,
                    'strand': strand,
                    'sequence': right_soft_clip_seq,
                    'clip_type': 'end' if read_strand == '+' else 'start',
                    'read_strand': read_strand,
                    'read_position': read.reference_end,
                    'distance_to_site': abs(read.reference_end - pos),
                    'sequence_length': len(right_soft_clip_seq),
                    'mapping_quality': read.mapping_quality
                }
                sequences.append(seq_info)
    
    return sequences


def extract_all_soft_clip_sequences(bam, candidate_sites, site_type, cfg):
    """
    Extract soft-clipped sequences from all candidate sites.
    
    Args:
        bam: pysam.AlignmentFile object
        candidate_sites: list of (chrom, pos, strand) tuples
        site_type: 'TSS' or 'TES'
        cfg: configuration object
        
    Returns:
        pandas.DataFrame with soft-clipped sequence information
    """
    all_sequences = []
    
    print(f"Extracting {site_type} soft-clipped sequences...")
    for idx, (chrom, pos, strand) in enumerate(candidate_sites):
        sequences = extract_soft_clip_sequences(bam, chrom, pos, strand, cfg)
        all_sequences.extend(sequences)
        
        if idx % 500 == 0:
            print(f"Processed {idx} {site_type} candidate sites, found {len(all_sequences)} sequences so far")
            sys.stdout.flush()
    
    print(f"Total {site_type} soft-clipped sequences extracted: {len(all_sequences)}")
    
    if not all_sequences:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'chrom', 'position', 'strand', 'sequence', 'clip_type', 
            'read_strand', 'read_position', 'distance_to_site', 
            'sequence_length', 'mapping_quality'
        ])
    
    return pd.DataFrame(all_sequences)


def main(cfg, config_path):
    """Main function to extract soft-clipped sequences."""
    if cfg is None:
        raise ValueError("Configuration has not been created. Please call create_config() first.")
    
    # Generate candidate files if needed
    cov_file, candidate_file, ref_candidate_file = gen_baseline_labels.main(cfg)
    cfg.set_cov_file(cov_file)
    cfg.set_candidate_file(candidate_file)
    cfg.set_ref_candidate_file(ref_candidate_file)

    # Open BAM file
    bam = pysam.AlignmentFile(cfg.bam_file, "rb")
    
    # Load candidate sites
    tss_candidate_sites, tes_candidate_sites = load_candidate_sites(cfg.candidate_file)
    
    # Extract TSS soft-clipped sequences
    tss_sequences_df = extract_all_soft_clip_sequences(bam, tss_candidate_sites, "TSS", cfg)
    tss_sequences_df.to_csv(cfg.tss_softclip_file, index=False)
    print(f"TSS soft-clipped sequences saved to: {cfg.tss_softclip_file}")
    
    # Extract TES soft-clipped sequences
    tes_sequences_df = extract_all_soft_clip_sequences(bam, tes_candidate_sites, "TES", cfg)
    tes_sequences_df.to_csv(cfg.tes_softclip_file, index=False)
    print(f"TES soft-clipped sequences saved to: {cfg.tes_softclip_file}")
    
    # Close BAM file
    bam.close()
    
    # Save configuration
    save_config(config_path)
    
    print("Soft-clipped sequence extraction complete!")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"TSS sequences: {len(tss_sequences_df)}")
    if len(tss_sequences_df) > 0:
        print(f"  - Start clips: {sum(tss_sequences_df['clip_type'] == 'start')}")
        print(f"  - End clips: {sum(tss_sequences_df['clip_type'] == 'end')}")
        print(f"  - Average sequence length: {tss_sequences_df['sequence_length'].mean():.1f}")
    
    print(f"TES sequences: {len(tes_sequences_df)}")
    if len(tes_sequences_df) > 0:
        print(f"  - Start clips: {sum(tes_sequences_df['clip_type'] == 'start')}")
        print(f"  - End clips: {sum(tes_sequences_df['clip_type'] == 'end')}")
        print(f"  - Average sequence length: {tes_sequences_df['sequence_length'].mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract soft-clipped sequences from BAM file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.config)