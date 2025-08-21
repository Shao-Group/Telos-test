import argparse
import pysam
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from config import load_config, save_config
import math
import sys
import re
import gc
import os
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import gen_baseline_labels as gen_baseline_labels
# from sequence_embeddings import get_embedder, cleanup_embedder
from kmer_embeddings import get_separate_kmer_embedders, cleanup_kmer_embedders


def load_candidate_sites(file_path):
    with open(file_path, 'r') as f:
        candidate_sites_text = f.read()
        # Parse the candidate sites
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



def extract_features(bam, chrom, pos, strand, cfg):
    region_start = max(0, pos - cfg.window_size)
    region_end = pos + cfg.window_size
    reads = bam.fetch(chrom, region_start, region_end)

    read_starts, read_ends, start_soft_clips, end_soft_clips, map_quals = [], [], [], [], []
    start_clip_sequences, end_clip_sequences = [], []
    strand_count = Counter()
    total_reads = 0

    for read in reads:
        if read.is_unmapped or read.mapping_quality < cfg.min_mapq:
            continue
        total_reads += 1
        read_strand = compute_strand(read)
        left_soft_clip_length = 0
        right_soft_clip_length = 0
        left_soft_clip_seq = ""
        right_soft_clip_seq = ""
        
        if read.cigartuples and read.query_sequence:
            if read.cigartuples[0][0] == 4:  # Soft clip at start
                left_soft_clip_length = read.cigartuples[0][1]
                left_soft_clip_seq = read.query_sequence[:left_soft_clip_length]
            if read.cigartuples[-1][0] == 4:  # Soft clip at end
                right_soft_clip_length = read.cigartuples[-1][1]
                right_soft_clip_seq = read.query_sequence[-right_soft_clip_length:]

        if read_strand == "+":
            read_starts.append(read.reference_start)
            read_ends.append(read.reference_end)
            start_soft_clips.append(left_soft_clip_length)
            end_soft_clips.append(right_soft_clip_length)
            if left_soft_clip_seq != "" and abs(read.reference_start - pos) <= cfg.soft_clip_window:
                start_clip_sequences.append(left_soft_clip_seq)
            if right_soft_clip_seq != "" and abs(read.reference_end - pos) <= cfg.soft_clip_window:
                end_clip_sequences.append(right_soft_clip_seq)
            
        else:
            read_starts.append(read.reference_end)
            read_ends.append(read.reference_start)
            start_soft_clips.append(right_soft_clip_length)
            end_soft_clips.append(left_soft_clip_length)
            if right_soft_clip_seq != "" and abs(read.reference_end - pos) <= cfg.soft_clip_window:
                start_clip_sequences.append(right_soft_clip_seq)
            if left_soft_clip_seq != "" and abs(read.reference_start - pos) <= cfg.soft_clip_window:
                end_clip_sequences.append(left_soft_clip_seq)

        map_quals.append(read.mapping_quality)
        if read_strand == strand:
            strand_count["forward"] += 1
        else:
            strand_count["reverse"] += 1

    

    window_radius = cfg.density_window
    read_start_density = sum(1 for s in read_starts if abs(s - pos) <= window_radius)
    read_end_density   = sum(1 for e in read_ends if abs(e - pos) <= window_radius)

    
    coverage_before, coverage_after, delta_coverage = calculate_coverage_change(bam, chrom, pos, cfg, strand)
    nearest_splice = find_nearest_splice_site(bam, chrom, pos, cfg)
    soft_clip_bias_value = soft_clip_bias(bam, chrom, pos, cfg, strand)
    start_entropy, end_entropy = read_start_end_entropy(read_starts, read_ends, pos, cfg)
    
    # Use soft-clipped sequences collected during main loop
    all_clips = start_clip_sequences + end_clip_sequences
    
    # Base features
    features = {
        "chrom": chrom,
        "position": pos,
        "strand": strand,
        "total_reads": total_reads,
        "read_start_density": read_start_density,
        "read_end_density": read_end_density,
        "start_soft_clip_mean": np.mean(start_soft_clips) if start_soft_clips else 0,
        "end_soft_clip_mean": np.mean(end_soft_clips) if end_soft_clips else 0,
        "start_soft_clip_max": max(start_soft_clips) if start_soft_clips else 0,
        "end_soft_clip_max": max(end_soft_clips) if end_soft_clips else 0,
        "start_soft_clip_median": np.median(start_soft_clips) if start_soft_clips else 0,
        "end_soft_clip_median": np.median(end_soft_clips) if end_soft_clips else 0,
        "start_soft_clip_count": len(start_soft_clips),
        "end_soft_clip_count": len(end_soft_clips),
        "mean_mapq": np.mean(map_quals) if map_quals else 0,
        "std_mapq": np.std(map_quals) if map_quals else 0,
        "strand_ratio": strand_count["forward"] / max(strand_count["reverse"], 1),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "delta_coverage": delta_coverage,
        "nearest_splice_dist": nearest_splice,
        "softclip_bias": soft_clip_bias_value,
        "start_entropy": start_entropy,
        "end_entropy": end_entropy, 
        "full_length_reads": sum(1 for r in reads if r.reference_start < region_start + 10 and r.reference_end > region_end - 10)
    }
    
    # Add sequence-based features (embeddings or handcrafted)
    if hasattr(cfg, 'use_embeddings') and cfg.use_embeddings:
        if hasattr(cfg, 'embedding_mode') and cfg.embedding_mode == 'hybrid':
            # Use hybrid approach: embeddings + key handcrafted features
            sequence_features = extract_hybrid_features(start_clip_sequences, end_clip_sequences, cfg)
        else:
            # Use pure embeddings approach
            raise NotImplementedError("Pure embeddings approach is not implemented yet")
            # sequence_features = extract_sequence_embeddings(start_clip_sequences, end_clip_sequences, cfg)
        features.update(sequence_features)
    else:
        # Use comprehensive handcrafted features with combined analysis only (better performance)
        # Determine site type from config
        current_site_type = getattr(cfg, 'current_site_type', 'TSS')
        combined_kmer_features = softclip_kmer_features(
            start_clip_sequences + end_clip_sequences, '', site_type=current_site_type
        )
        features.update(combined_kmer_features)
    
    # Ensure absolute order consistency by sorting feature names
    # Keep basic features first, then sort pattern-based features
    basic_features = ['chrom', 'position', 'strand', 'total_reads', 'read_start_density', 
                     'read_end_density', 'start_soft_clip_mean', 'end_soft_clip_mean',
                     'start_soft_clip_max', 'end_soft_clip_max', 'start_soft_clip_median',
                     'end_soft_clip_median', 'start_soft_clip_count', 'end_soft_clip_count',
                     'mean_mapq', 'std_mapq', 'strand_ratio', 'coverage_before',
                     'coverage_after', 'delta_coverage', 'nearest_splice_dist',
                     'softclip_bias', 'start_entropy', 'end_entropy', 'full_length_reads']
    
    ordered_features = OrderedDict()
    # Add basic features in predefined order
    for key in basic_features:
        if key in features:
            ordered_features[key] = features[key]
    
    # Add remaining features in alphabetical order for consistency
    remaining_keys = sorted([k for k in features.keys() if k not in basic_features])
    for key in remaining_keys:
        ordered_features[key] = features[key]
    
    return dict(ordered_features)


def calculate_coverage_change(bam, chrom, pos, cfg, strand):
    """Calculate coverage before and after the candidate site."""
    if pos < cfg.coverage_window:
        print(f"Warning: Position {pos} is too close to the start of the chromosome {chrom}.")
        coverage_before = 0
    else:
        region_before = bam.count_coverage(
            chrom, pos - cfg.coverage_window, pos, quality_threshold=cfg.min_mapq)
        coverage_before = int(np.sum(region_before))
    
    chrom_length = bam.get_reference_length(chrom)
    if pos + cfg.coverage_window > chrom_length:
        print(f"Warning: Position {pos} is too close to the end of the chromosome {chrom}.")
        coverage_after = 0
    else:
        # Ensure we don't go out of bounds
        region_after = bam.count_coverage(
            chrom, pos, pos + cfg.coverage_window, quality_threshold=cfg.min_mapq)
        coverage_after = int(np.sum(region_after))

    delta_coverage = coverage_after - coverage_before
    if strand == "+":
        return coverage_before, coverage_after, delta_coverage
    else:
        return coverage_after, coverage_before, -delta_coverage

def find_nearest_splice_site(bam, chrom, pos, cfg):
    """Find distance to the nearest splice junction (N CIGAR operation)."""
    min_distance = cfg.splice_site_window
    for read in bam.fetch(chrom, max(0, pos - cfg.splice_site_window), pos + cfg.splice_site_window):
        if read.is_unmapped or read.mapping_quality < cfg.min_mapq or not read.cigartuples:
            continue
        curr_pos = read.reference_start
        for op, length in read.cigartuples:
            if op == 0 or op == 7 or op == 8:  # Match/Mismatch (M/=/X)
                curr_pos += length
            elif op == 3:  # 'N' = spliced region
                splice_site_start = curr_pos
                splice_site_end = curr_pos + length
                distance = min(abs(pos - splice_site_start), abs(pos - splice_site_end))
                min_distance = min(min_distance, distance)
                curr_pos += length
            else:
                curr_pos += length
    return min_distance

def soft_clip_bias(bam, chrom, pos, cfg, strand):
    """Ratio of reads with soft-clip starting exactly at the site."""
    soft_clip_count = 0
    total_reads = 0
    for read in bam.fetch(chrom, max(0, pos - cfg.soft_clip_window), pos + cfg.soft_clip_window):
        read_strand = compute_strand(read)
        if read.is_unmapped or read.mapping_quality < cfg.min_mapq or not read.cigartuples or read_strand != strand:
            continue
        total_reads += 1
        if read.cigartuples[0][0] == 4 and abs(read.reference_start - pos) <= 5:
            soft_clip_count += 1
        elif read.cigartuples[-1][0] == 4 and abs(read.reference_end - pos) <= 5:
            soft_clip_count += 1
    return soft_clip_count / total_reads if total_reads else 0


def calculate_entropy(positions):
    """Calculate entropy of read starts or ends."""
    if len(positions) == 0:
        return 0
    count = Counter(positions)
    total = sum(count.values())
    entropy = -sum((freq/total) * math.log2(freq/total) for freq in count.values())
    return entropy



def softclip_kmer_features(clipped_sequences, prefix="", klist=[3], site_type="TSS"):
    """Enhanced comprehensive soft-clip feature extraction with positional bias and biological motifs."""
    if not clipped_sequences:
        clipped_sequences = [""]
    combined = {}
    
    # Single pass through sequences to extract all features efficiently
    all_bases = ''.join(clipped_sequences).upper()
    total_bases = len(all_bases)
    
    # 1. Nucleotide composition (single pass)
    if total_bases > 0:
        a_count = all_bases.count('A')
        t_count = all_bases.count('T') 
        g_count = all_bases.count('G')
        c_count = all_bases.count('C')
        
        combined[f'{prefix}gc_content'] = (g_count + c_count) / total_bases
        combined[f'{prefix}purine_ratio'] = (a_count + g_count) / total_bases
    else:
        combined[f'{prefix}gc_content'] = 0
        combined[f'{prefix}purine_ratio'] = 0
    
    # 2. Homopolymer features (single pass per base type)
    bases = ['A', 'C', 'G', 'T']
    for base in bases:
        max_run = 0
        for seq in clipped_sequences:
            seq = seq.upper()
            matches = re.findall(f'{base}+', seq)
            if matches:
                seq_max = max(len(match) for match in matches)
                max_run = max(max_run, seq_max)
        combined[f'{prefix}max_poly{base}'] = max_run
    
    # 3. Enhanced motif features with biological relevance
    combined.update(_extract_enhanced_motif_features(clipped_sequences, prefix, site_type))
    
    # 4. Enhanced K-mer features with positional bias
    for k in klist:
        k_features = _extract_enhanced_kmer_features(clipped_sequences, prefix, k)
        combined.update(k_features)
    
    # 5. Length-stratified features
    # combined.update(_extract_length_stratified_features(clipped_sequences, prefix))
    
    return combined

def _extract_motif_features_optimized(clipped_sequences, prefix):
    """Optimized motif detection in single pass."""
    motif_counts = {'cg_rich': 0, 'polyA_signals': 0, 'splice_signals': 0, 'tata_like': 0}
    
    for seq in clipped_sequences:
        seq_upper = seq.upper()
        if 'AAAA' in seq_upper:
            motif_counts['polyA_signals'] += 1
        if 'TATA' in seq_upper:
            motif_counts['tata_like'] += 1
        if 'GT' in seq_upper or 'AG' in seq_upper:
            motif_counts['splice_signals'] += 1
        if seq_upper.count('CG') >= 2:
            motif_counts['cg_rich'] += 1
    return {f'{prefix}{name}': count for name, count in motif_counts.items()}

def _extract_enhanced_motif_features(clipped_sequences, prefix, site_type):
    """Enhanced biological motif detection with site-specific relevance."""
    motif_counts = {}
    
    # Define site-specific motifs
    if site_type.upper() == 'TSS':
        # TSS-relevant motifs
        splice_motifs = ['GT', 'GC', 'AG', 'AT']  # splice donor/acceptor
        core_promoter = ['TATA', 'CAAT', 'GC']  # core promoter elements
        cap_signals = ['CT', 'GA', 'CA']  # cap site signals
        motif_sets = {
            'splice_donors': ['GTA', 'GTG', 'GTC', 'GTT'],
            'splice_acceptors': ['CAG', 'TAG', 'AAG', 'GAG'],
            'promoter_signals': ['TATA', 'CAAT'],
            'cap_site_signals': ['CT', 'GA', 'CA'],
            'purine_rich': ['AAG', 'GAA', 'AGA', 'GAG']
        }
    else:  # TES
        # TES-relevant motifs
        polya_variants = ['AATAA', 'ATTAA', 'AATACA', 'AATAGA', 'AATAAA']
        cleavage_signals = ['CA', 'GT', 'GG']  # cleavage and polyadenylation
        downstream_elements = ['TGT', 'GGT', 'GTG']
        motif_sets = {
            'polya_signals': polya_variants,
            'cleavage_sites': ['CA', 'GT', 'GG'],
            'downstream_elements': ['TGT', 'GGT', 'GTG'],
            'u_rich_regions': ['TTT', 'TTA', 'TAT', 'ATT'],
            'pyrimidine_rich': ['TTC', 'CTT', 'CCT', 'TCC']
        }
    
    # Count motifs with positional awareness
    for motif_type, motifs in motif_sets.items():
        total_count = 0
        terminus_count = 0  # Count in terminal 5bp
        
        for seq in clipped_sequences:
            seq_upper = seq.upper()
            seq_len = len(seq_upper)
            
            for motif in motifs:
                # Total count
                motif_count = seq_upper.count(motif)
                total_count += motif_count
                
                # Terminal enrichment (first/last 5bp)
                if seq_len >= len(motif):
                    # Check terminals
                    terminal_5bp_start = seq_upper[:min(5, seq_len)]
                    terminal_5bp_end = seq_upper[max(0, seq_len-5):]
                    terminus_count += terminal_5bp_start.count(motif)
                    terminus_count += terminal_5bp_end.count(motif)
        
        motif_counts[f'{prefix}{motif_type}_total'] = total_count
        motif_counts[f'{prefix}{motif_type}_terminus'] = terminus_count
        
        # Calculate terminal enrichment ratio
        if total_count > 0:
            motif_counts[f'{prefix}{motif_type}_terminal_ratio'] = terminus_count / total_count
        else:
            motif_counts[f'{prefix}{motif_type}_terminal_ratio'] = 0
    
    # Degenerate motif matching (1 mismatch allowed for key motifs)
    key_motifs = ['AATAAA'] if site_type.upper() == 'TES' else ['TATAAA']
    for motif in key_motifs:
        degenerate_count = 0
        for seq in clipped_sequences:
            degenerate_count += _count_degenerate_motif(seq.upper(), motif, max_mismatches=1)
        motif_counts[f'{prefix}degenerate_{motif.lower()}'] = degenerate_count
    
    return motif_counts

def _count_degenerate_motif(sequence, motif, max_mismatches=1):
    """Count occurrences of motif allowing up to max_mismatches."""
    count = 0
    motif_len = len(motif)
    
    for i in range(len(sequence) - motif_len + 1):
        subseq = sequence[i:i + motif_len]
        mismatches = sum(1 for a, b in zip(subseq, motif) if a != b)
        if mismatches <= max_mismatches:
            count += 1
    
    return count

def _extract_kmer_features_optimized(clipped_sequences, prefix, k):
    """Optimized k-mer feature extraction in single pass."""
    # Fixed k-mers for biological relevance (only 3-mers for efficiency)
    if k == 3:
        fixed_kmers = sorted(['ATG', 'TAA', 'TGA', 'TAG', 'GCC', 'CGC', 'GCG', 
                             'AAA', 'TTT', 'AAT', 'GTG', 'CAG', 'GAG', 
                             'CCC', 'GGG', 'CCG'])
    else:
        fixed_kmers = []  # Only use 3-mers for fixed vocabulary to reduce redundancy
    
    # Single pass through sequences to collect all k-mer data
    kmer_counts = Counter()
    fixed_kmer_counts = Counter()
    composition_counts = {'at_rich_kmers': 0, 'balanced_kmers': 0, 'gc_rich_kmers': 0, 
                         'purine_rich_kmers': 0, 'repeat_kmers': 0}
    total_kmers = 0
    gc_kmers = 0
    
    for seq in clipped_sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(base in 'ATGC' for base in kmer):
                kmer_counts[kmer] += 1
                total_kmers += 1
                
                # Fixed k-mer counting
                if kmer in fixed_kmers:
                    fixed_kmer_counts[kmer] += 1
                
                # GC content analysis
                gc_content = (kmer.count('G') + kmer.count('C')) / k
                if gc_content >= 0.5:
                    gc_kmers += 1
                
                # Composition analysis
                purine_content = (kmer.count('A') + kmer.count('G')) / k
                if len(set(kmer)) == 1:
                    composition_counts['repeat_kmers'] += 1
                elif gc_content > 0.6:
                    composition_counts['gc_rich_kmers'] += 1
                elif gc_content < 0.4:
                    composition_counts['at_rich_kmers'] += 1
                else:
                    composition_counts['balanced_kmers'] += 1
                
                if purine_content > 0.6:
                    composition_counts['purine_rich_kmers'] += 1
    
    # Compile features
    features = {}
    
    # Fixed k-mer features (only for 3-mers)
    if k == 3:
        for kmer in fixed_kmers:
            features[f'{prefix}kmer_{kmer}'] = fixed_kmer_counts.get(kmer, 0)
    
    # Statistical features
    if total_kmers > 0:
        unique_kmers = len(kmer_counts)
        diversity = unique_kmers / total_kmers
        most_frequent_count = max(kmer_counts.values()) if kmer_counts else 0
        most_frequent_ratio = most_frequent_count / total_kmers
        
        # Shannon entropy
        probs = [count / total_kmers for count in kmer_counts.values()]
        kmer_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        features.update({
            f'{prefix}k{k}_gc_kmers_ratio': gc_kmers / total_kmers,
            f'{prefix}k{k}_kmer_diversity': diversity,
            f'{prefix}k{k}_kmer_entropy': kmer_entropy,
            f'{prefix}k{k}_most_frequent_kmer_ratio': most_frequent_ratio,
            f'{prefix}k{k}_unique_kmers': unique_kmers
        })
        
        # Composition features
        for comp_type, count in composition_counts.items():
            features[f'{prefix}k{k}_{comp_type}'] = count
    else:
        # Zero features when no k-mers found
        zero_features = [f'{prefix}k{k}_gc_kmers_ratio', f'{prefix}k{k}_kmer_diversity', 
                        f'{prefix}k{k}_kmer_entropy', f'{prefix}k{k}_most_frequent_kmer_ratio',
                        f'{prefix}k{k}_unique_kmers']
        for comp_type in composition_counts.keys():
            zero_features.append(f'{prefix}k{k}_{comp_type}')
        features.update({feat: 0 for feat in zero_features})
    
    return features

def _extract_enhanced_kmer_features(clipped_sequences, prefix, k):
    """Enhanced k-mer feature extraction with positional bias."""
    # Get standard k-mer features first
    standard_features = _extract_kmer_features_optimized(clipped_sequences, prefix, k)
    
    # Add positional bias features
    positional_features = {}
    
    # Positional k-mer counts
    fiveprime_kmers = Counter()
    threeprime_kmers = Counter()
    middle_kmers = Counter()
    
    total_positional_kmers = 0
    
    for seq in clipped_sequences:
        seq = seq.upper()
        seq_len = len(seq)
        
        if seq_len < k:
            continue
            
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(base in 'ATGC' for base in kmer):
                total_positional_kmers += 1
                
                # Calculate relative position (0 to 1)
                position_ratio = i / max(1, seq_len - k)
                
                # Bin positions
                if position_ratio <= 0.3:  # 5' region
                    fiveprime_kmers[kmer] += 1
                elif position_ratio >= 0.7:  # 3' region
                    threeprime_kmers[kmer] += 1
                else:  # Middle region
                    middle_kmers[kmer] += 1
    
    # Calculate positional bias statistics
    if total_positional_kmers > 0:
        fiveprime_total = sum(fiveprime_kmers.values())
        threeprime_total = sum(threeprime_kmers.values())
        middle_total = sum(middle_kmers.values())
        
        positional_features.update({
            f'{prefix}k{k}_5prime_ratio': fiveprime_total / total_positional_kmers,
            f'{prefix}k{k}_3prime_ratio': threeprime_total / total_positional_kmers,
            f'{prefix}k{k}_middle_ratio': middle_total / total_positional_kmers,
            f'{prefix}k{k}_5prime_diversity': len(fiveprime_kmers) / max(1, fiveprime_total),
            f'{prefix}k{k}_3prime_diversity': len(threeprime_kmers) / max(1, threeprime_total),
        })
        
        # Most frequent k-mers in each region
        if fiveprime_kmers:
            most_freq_5prime = max(fiveprime_kmers.values())
            positional_features[f'{prefix}k{k}_5prime_max_freq'] = most_freq_5prime / max(1, fiveprime_total)
        else:
            positional_features[f'{prefix}k{k}_5prime_max_freq'] = 0
            
        if threeprime_kmers:
            most_freq_3prime = max(threeprime_kmers.values())
            positional_features[f'{prefix}k{k}_3prime_max_freq'] = most_freq_3prime / max(1, threeprime_total)
        else:
            positional_features[f'{prefix}k{k}_3prime_max_freq'] = 0
            
        # Terminal enrichment for specific biological k-mers (3-mers only)
        if k == 3:
            important_kmers = ['ATG', 'TAA', 'TGA', 'TAG', 'AAA', 'TTT']
            for kmer in important_kmers:
                total_kmer = fiveprime_kmers.get(kmer, 0) + threeprime_kmers.get(kmer, 0) + middle_kmers.get(kmer, 0)
                terminal_kmer = fiveprime_kmers.get(kmer, 0) + threeprime_kmers.get(kmer, 0)
                
                if total_kmer > 0:
                    positional_features[f'{prefix}k{k}_{kmer}_terminal_enrichment'] = terminal_kmer / total_kmer
                else:
                    positional_features[f'{prefix}k{k}_{kmer}_terminal_enrichment'] = 0
    else:
        # Zero features when no k-mers found
        zero_positional_features = [
            f'{prefix}k{k}_5prime_ratio', f'{prefix}k{k}_3prime_ratio', f'{prefix}k{k}_middle_ratio',
            f'{prefix}k{k}_5prime_diversity', f'{prefix}k{k}_3prime_diversity',
            f'{prefix}k{k}_5prime_max_freq', f'{prefix}k{k}_3prime_max_freq'
        ]
        
        if k == 3:
            important_kmers = ['ATG', 'TAA', 'TGA', 'TAG', 'AAA', 'TTT']
            for kmer in important_kmers:
                zero_positional_features.append(f'{prefix}k{k}_{kmer}_terminal_enrichment')
        
        positional_features.update({feat: 0 for feat in zero_positional_features})
    
    # Combine standard and positional features
    combined_features = {**standard_features, **positional_features}
    return combined_features

def _extract_length_stratified_features(clipped_sequences, prefix):
    """Extract features based on clip length stratification."""
    features = {}
    
    # Stratify sequences by length
    short_clips = [seq for seq in clipped_sequences if len(seq) <= 10]
    medium_clips = [seq for seq in clipped_sequences if 11 <= len(seq) <= 25]
    long_clips = [seq for seq in clipped_sequences if len(seq) > 25]
    
    # Basic statistics for each stratum
    strata = {
        'short': short_clips,
        'medium': medium_clips,
        'long': long_clips
    }
    
    for stratum_name, clips in strata.items():
        if clips:
            # Length statistics
            lengths = [len(seq) for seq in clips]
            features[f'{prefix}{stratum_name}_count'] = len(clips)
            features[f'{prefix}{stratum_name}_mean_length'] = np.mean(lengths)
            features[f'{prefix}{stratum_name}_max_length'] = max(lengths)
            
            # Composition for each stratum
            all_bases = ''.join(clips).upper()
            if all_bases:
                total_bases = len(all_bases)
                features[f'{prefix}{stratum_name}_gc_content'] = (all_bases.count('G') + all_bases.count('C')) / total_bases
                features[f'{prefix}{stratum_name}_purine_ratio'] = (all_bases.count('A') + all_bases.count('G')) / total_bases
            else:
                features[f'{prefix}{stratum_name}_gc_content'] = 0
                features[f'{prefix}{stratum_name}_purine_ratio'] = 0
        else:
            # Zero features for empty strata
            features[f'{prefix}{stratum_name}_count'] = 0
            features[f'{prefix}{stratum_name}_mean_length'] = 0
            features[f'{prefix}{stratum_name}_max_length'] = 0
            features[f'{prefix}{stratum_name}_gc_content'] = 0
            features[f'{prefix}{stratum_name}_purine_ratio'] = 0
    
    # Cross-stratum ratios
    total_clips = len(clipped_sequences)
    if total_clips > 0:
        features[f'{prefix}short_clip_ratio'] = len(short_clips) / total_clips
        features[f'{prefix}medium_clip_ratio'] = len(medium_clips) / total_clips
        features[f'{prefix}long_clip_ratio'] = len(long_clips) / total_clips
    else:
        features[f'{prefix}short_clip_ratio'] = 0
        features[f'{prefix}medium_clip_ratio'] = 0
        features[f'{prefix}long_clip_ratio'] = 0
    
    return features








def extract_kmer_embeddings(start_clip_sequences, end_clip_sequences, cfg, site_type):
    """Extract k-mer embeddings from soft-clipped sequences."""
    try:
        # Get the appropriate k-mer embedders
        start_tss, start_tes, end_tss, end_tes = get_separate_kmer_embedders()
        
        # Determine which embedders to use based on site type
        if site_type.upper() == 'TSS':
            start_embedder = start_tss
            end_embedder = end_tss
        elif site_type.upper() == 'TES':
            start_embedder = start_tes
            end_embedder = end_tes
        else:
            # Default to TSS embedders
            start_embedder = start_tss
            end_embedder = end_tss
            print(f"Warning: Unknown site_type '{site_type}', using TSS embedders")
        
        # Generate embeddings for start and end clips
        start_embedding = start_embedder.embed_site(start_clip_sequences)
        end_embedding = end_embedder.embed_site(end_clip_sequences)
        
        # Create feature dictionary
        features = {}
        
        # Add start k-mer embeddings
        start_feature_names = start_embedder.get_feature_names()
        for i, (name, val) in enumerate(zip(start_feature_names, start_embedding)):
            features[f'start_kmer_{name}'] = float(val)
        
        # Add end k-mer embeddings  
        end_feature_names = end_embedder.get_feature_names()
        for i, (name, val) in enumerate(zip(end_feature_names, end_embedding)):
            features[f'end_kmer_{name}'] = float(val)
        
        # Add combined statistics
        combined_embedding = (start_embedding + end_embedding) / 2
        for i, val in enumerate(combined_embedding):
            features[f'combined_kmer_{i}'] = float(val)
        
        return features
        
    except Exception as e:
        print(f"Warning: Could not extract k-mer embeddings: {e}")
        # Return empty features on error
        return {}


# def extract_sequence_embeddings(start_clip_sequences, end_clip_sequences, cfg):
#     """Extract learned embeddings from soft-clipped sequences using separate models."""
#     # Determine embedding type
#     embedding_type = getattr(cfg, 'embedding_type', 'cnn')
#     embedding_dim = getattr(cfg, 'embedding_dim', 128)
    
#     if embedding_type == 'kmer':
#         # Use k-mer embeddings
#         site_type = getattr(cfg, 'current_site_type', 'TSS')  # Will be set during extraction
#         return extract_kmer_embeddings(start_clip_sequences, end_clip_sequences, cfg, site_type)
#     elif embedding_type == 'cnn':
#         # Use CNN embeddings with separate models for start and end clips
#         try:
#             from sequence_cnn_embeddings import get_separate_cnn_embedders
            
#             # Get model paths for start and end clips
#             start_model_path = getattr(cfg, 'cnn_start_model_path', None)
#             end_model_path = getattr(cfg, 'cnn_end_model_path', None)
            
#             # Try to use separate models first
#             if start_model_path and end_model_path:
#                 start_embedder, end_embedder = get_separate_cnn_embedders(start_model_path, end_model_path)
#                 print("Separate CNN embedders loaded successfully")
                
#                 # Get embeddings using separate models
#                 start_embedding = start_embedder.aggregate_embeddings(start_clip_sequences, method="mean")
#                 end_embedding = end_embedder.aggregate_embeddings(end_clip_sequences, method="mean")
                
#                 # Combined embedding (average of start and end embeddings)
#                 combined_embedding = (start_embedding + end_embedding) / 2
                
#             else:
#                 # Fallback to single model if separate models not available
#                 print("Separate models not found, falling back to single CNN model")
#                 from sequence_cnn_embeddings import get_cnn_embedder
#                 model_path = getattr(cfg, 'cnn_model_path', None)
#                 embedder = get_cnn_embedder(model_path)
                
#                 start_embedding = embedder.aggregate_embeddings(start_clip_sequences, method="mean")
#                 end_embedding = embedder.aggregate_embeddings(end_clip_sequences, method="mean")
                
#                 # Combined embedding from all clips
#                 all_clips = start_clip_sequences + end_clip_sequences
#                 combined_embedding = embedder.aggregate_embeddings(all_clips, method="mean")
                
#         except Exception as e:
#             print(f"Warning: Could not load CNN embedder: {e}")
#             print("Falling back to zero embeddings")
#             features = {}
#             for i in range(embedding_dim):
#                 features[f'start_embed_{i}'] = 0.0
#                 features[f'end_embed_{i}'] = 0.0
#                 features[f'combined_embed_{i}'] = 0.0
#             return features
#     else:
#         # Use DNABERT embeddings (fallback)
#         try:
#             embedder = get_embedder()
#             embedding_dim = 768  # DNABERT embedding dimension
            
#             start_embedding = embedder.aggregate_embeddings(start_clip_sequences, method="mean")
#             end_embedding = embedder.aggregate_embeddings(end_clip_sequences, method="mean")
            
#             # Combined embedding from all clips
#             all_clips = start_clip_sequences + end_clip_sequences
#             combined_embedding = embedder.aggregate_embeddings(all_clips, method="mean")
            
#         except Exception as e:
#             print(f"Warning: Could not load DNABERT embedder: {e}")
#             print("Falling back to zero embeddings")
#             features = {}
#             for i in range(768):
#                 features[f'start_embed_{i}'] = 0.0
#                 features[f'end_embed_{i}'] = 0.0
#                 features[f'combined_embed_{i}'] = 0.0
#             return features
    
#     # Create feature dictionary with consistent naming
#     try:
#         features = {}
        
#         # Add start embeddings
#         for i, val in enumerate(start_embedding):
#             features[f'start_embed_{i}'] = float(val)
            
#         # Add end embeddings  
#         for i, val in enumerate(end_embedding):
#             features[f'end_embed_{i}'] = float(val)
            
#         # Add combined embeddings
#         for i, val in enumerate(combined_embedding):
#             features[f'combined_embed_{i}'] = float(val)
            
#         return features
        
#     except Exception as e:
#         print(f"Warning: Error in embedding extraction: {e}")
#         # Return zero embeddings on error
#         features = {}
#         for i in range(embedding_dim):
#             features[f'start_embed_{i}'] = 0.0
#             features[f'end_embed_{i}'] = 0.0
#             features[f'combined_embed_{i}'] = 0.0
#         return features


def extract_hybrid_features(start_clip_sequences, end_clip_sequences, cfg):
    """Extract both embeddings and key handcrafted features."""
    # Get embeddings (could be CNN or k-mer based on cfg.embedding_type)
    # embedding_features = extract_kmer_embeddings(start_clip_sequences, end_clip_sequences, cfg, cfg.current_site_type)
    # Use only combined features for better performance
    current_site_type = getattr(cfg, 'current_site_type', 'TSS')
    combined_kmer_features = softclip_kmer_features(
        start_clip_sequences + end_clip_sequences, '', site_type=current_site_type
    )

    embedding_features = combined_kmer_features
    # Keep only the most important handcrafted features
    all_clips = start_clip_sequences + end_clip_sequences
    
    # Basic sequence statistics
    total_clip_length = sum(len(seq) for seq in all_clips)
    avg_clip_length = np.mean([len(seq) for seq in all_clips]) if all_clips else 0
    max_clip_length = max(len(seq) for seq in all_clips) if all_clips else 0
    num_clips = len(all_clips)
    
    # Basic composition
    if all_clips:
        all_bases = ''.join(all_clips).upper()
        total_bases = len(all_bases)
        gc_content = (all_bases.count('G') + all_bases.count('C')) / total_bases if total_bases > 0 else 0
        at_content = (all_bases.count('A') + all_bases.count('T')) / total_bases if total_bases > 0 else 0
    else:
        gc_content = 0
        at_content = 0
    
    # Key handcrafted features
    basic_features = {
        'total_clip_length': total_clip_length,
        'avg_clip_length': avg_clip_length,
        'max_clip_length': max_clip_length,
        'num_clips': num_clips,
        'gc_content_basic': gc_content,
        'at_content_basic': at_content,
    }
    
    # Combine embeddings with basic features
    features = {}
    features.update(embedding_features)
    features.update(basic_features)
    
    return features


def compute_strand(read):
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

def read_start_end_entropy(start_positions, end_positions, pos, cfg):
    """Entropy around read start and end positions."""
    start_entropy = calculate_entropy(start_positions)
    end_entropy = calculate_entropy(end_positions)
    return start_entropy, end_entropy





def main(cfg, config_path):
    if cfg == None:
        raise ValueError("Configuration has not been created. Please call create_config() first.")
        return

    cov_file, candidate_file, ref_candidate_file = gen_baseline_labels.main(cfg)
    cfg.set_cov_file(cov_file)
    cfg.set_candidate_file(candidate_file)
    cfg.set_ref_candidate_file(ref_candidate_file)

    bam = pysam.AlignmentFile(cfg.bam_file, "rb")  # <-- adjust path if needed
    tss_candidate_sites, tes_candidate_sites = load_candidate_sites(cfg.candidate_file)
    
    # # Collect features
    if os.path.exists(cfg.tss_feature_file) and os.path.exists(cfg.tes_feature_file):
        print(f"Feature files already exist: {cfg.tss_feature_file}, {cfg.tes_feature_file}")
        # Close the BAM file
        bam.close()
        save_config(config_path)

        return

    print("Extracting TSS candidate features...")
    # Set current site type for k-mer embeddings
    cfg.current_site_type = 'TSS'
    
    # Process in batches to save memory
    batch_size = 1000
    total_sites = len(tss_candidate_sites)
    
    # Initialize CSV file with headers
    first_batch = True
    
    for batch_start in range(0, total_sites, batch_size):
        batch_end = min(batch_start + batch_size, total_sites)
        batch_sites = tss_candidate_sites[batch_start:batch_end]
        
        print(f"Processing TSS batch {batch_start//batch_size + 1}/{(total_sites-1)//batch_size + 1} ({batch_start}-{batch_end-1})")
        
        # Extract features for this batch
        batch_features = []
        for idx, site in enumerate(batch_sites):
            batch_features.append(extract_features(bam, *site, cfg))
            if (batch_start + idx) % 500 == 0:
                print(f"Extracted {batch_start + idx} TSS candidate features")
                sys.stdout.flush()
        
        # Save batch to CSV
        batch_df = pd.DataFrame(batch_features)
        if first_batch:
            batch_df.to_csv(cfg.tss_feature_file, index=False, mode='w')
            first_batch = False
        else:
            batch_df.to_csv(cfg.tss_feature_file, index=False, mode='a', header=False)
        
        # Clear memory
        del batch_features
        del batch_df
        gc.collect()  # Force garbage collection

    print("Extracting TES candidate features...")
    # Set current site type for k-mer embeddings
    cfg.current_site_type = 'TES'
    
    # Process in batches to save memory
    total_sites = len(tes_candidate_sites)
    
    # Initialize CSV file with headers
    first_batch = True
    
    for batch_start in range(0, total_sites, batch_size):
        batch_end = min(batch_start + batch_size, total_sites)
        batch_sites = tes_candidate_sites[batch_start:batch_end]
        
        print(f"Processing TES batch {batch_start//batch_size + 1}/{(total_sites-1)//batch_size + 1} ({batch_start}-{batch_end-1})")
        
        # Extract features for this batch
        batch_features = []
        for idx, site in enumerate(batch_sites):
            batch_features.append(extract_features(bam, *site, cfg))
            if (batch_start + idx) % 500 == 0:
                print(f"Extracted {batch_start + idx} TES candidate features")
                sys.stdout.flush()
        
        # Save batch to CSV
        batch_df = pd.DataFrame(batch_features)
        if first_batch:
            batch_df.to_csv(cfg.tes_feature_file, index=False, mode='w')
            first_batch = False
        else:
            batch_df.to_csv(cfg.tes_feature_file, index=False, mode='a', header=False)
        
        # Clear memory
        del batch_features
        del batch_df
        gc.collect()  # Force garbage collection
    
    # Clean up embedder to free GPU memory
    if hasattr(cfg, 'use_embeddings') and cfg.use_embeddings:
        embedding_type = getattr(cfg, 'embedding_type', 'kmer')
        if embedding_type == 'kmer':
            print("Cleaning up k-mer embedders...")
            cleanup_kmer_embedders()
    
    print("Feature extraction complete!")
    
    
    
    save_config(config_path)
    
    print("Feature extraction and selection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from BAM file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.config)