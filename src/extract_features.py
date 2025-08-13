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
from tqdm import tqdm
from scipy.stats import entropy
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
            if left_soft_clip_seq and abs(read.reference_start - pos) <= cfg.soft_clip_window:
                start_clip_sequences.append(left_soft_clip_seq)
            if right_soft_clip_seq and abs(read.reference_end - pos) <= cfg.soft_clip_window:
                end_clip_sequences.append(right_soft_clip_seq)
            
        else:
            read_starts.append(read.reference_end)
            read_ends.append(read.reference_start)
            start_soft_clips.append(right_soft_clip_length)
            end_soft_clips.append(left_soft_clip_length)
            if right_soft_clip_seq and abs(read.reference_end - pos) <= cfg.soft_clip_window:
                start_clip_sequences.append(right_soft_clip_seq)
            if left_soft_clip_seq and abs(read.reference_start - pos) <= cfg.soft_clip_window:
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
        # Use original handcrafted features
        kmer_features = softclip_kmer_features(all_clips)
        motif_features = softclip_motif_features(all_clips)
        start_comp_features = softclip_composition_features(start_clip_sequences, "start_")
        end_comp_features = softclip_composition_features(end_clip_sequences, "end_")
        start_homo_features = softclip_homopolymer_features(start_clip_sequences, "start_")
        end_homo_features = softclip_homopolymer_features(end_clip_sequences, "end_")
        
        # Add pattern-based features in consistent order
        # 1. K-mer features (fixed kmers + stats + composition)
        features.update(kmer_features)
        # 2. Motif features
        features.update(motif_features)
        # 3. Composition features (start, then end)
        features.update(start_comp_features)
        features.update(end_comp_features)
        # 4. Homopolymer features (start, then end)
        features.update(start_homo_features)
        features.update(end_homo_features)
    
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

def softclip_fixed_kmer_features(clipped_sequences, k=3):
    """Use a fixed set of biologically relevant k-mers."""
    # Define fixed vocabulary of important k-mers for TSS/TES (sorted for consistency)
    fixed_kmers = sorted([
        'ATG', 'TAA', 'TGA', 'TAG',  # Start/stop codons
        'GCC', 'CGC', 'GCG',        # GC-rich
        'AAA', 'TTT', 'AAT',        # AT-rich  
        'GTG', 'CAG', 'GAG',        # Common codons
        'CCC', 'GGG', 'CCG'         # Other patterns
    ])
    
    if not clipped_sequences:
        # Return in sorted order for consistency
        return {f'kmer_{kmer}': 0 for kmer in fixed_kmers}
    
    kmer_counts = Counter()
    for seq in clipped_sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in fixed_kmers:
                kmer_counts[kmer] += 1
    
    # Return in sorted order for consistency
    return {f'kmer_{kmer}': kmer_counts.get(kmer, 0) for kmer in fixed_kmers}

def softclip_kmer_stats(clipped_sequences, k=3):
    """Extract statistical features from k-mer distribution."""
    # Define feature names in consistent order
    feature_names = [
        'gc_kmers_ratio',
        'kmer_diversity', 
        'kmer_entropy',
        'most_frequent_kmer_ratio',
        'unique_kmers'
    ]
    
    if not clipped_sequences:
        return {name: 0 for name in feature_names}
    
    kmer_counts = Counter()
    total_kmers = 0
    gc_kmers = 0
    
    for seq in clipped_sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(base in 'ATGC' for base in kmer):  # Valid nucleotides only
                kmer_counts[kmer] += 1
                total_kmers += 1
                if (kmer.count('G') + kmer.count('C')) >= k/2:
                    gc_kmers += 1
    
    if total_kmers == 0:
        return {name: 0 for name in feature_names}
    
    # Calculate statistics
    unique_kmers = len(kmer_counts)
    diversity = unique_kmers / total_kmers if total_kmers > 0 else 0
    most_frequent_count = max(kmer_counts.values()) if kmer_counts else 0
    most_frequent_ratio = most_frequent_count / total_kmers if total_kmers > 0 else 0
    
    # Shannon entropy
    probs = [count / total_kmers for count in kmer_counts.values()]
    kmer_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    
    # Return in consistent order
    return {
        'gc_kmers_ratio': gc_kmers / total_kmers if total_kmers > 0 else 0,
        'kmer_diversity': diversity,
        'kmer_entropy': kmer_entropy,
        'most_frequent_kmer_ratio': most_frequent_ratio, 
        'unique_kmers': unique_kmers
    }

def softclip_kmer_composition(clipped_sequences, k=3):
    """Count k-mers by composition type."""
    # Define feature names in consistent order
    feature_names = [
        'at_rich_kmers',       # >60% AT  
        'balanced_kmers',      # 40-60% GC
        'gc_rich_kmers',       # >60% GC
        'purine_rich_kmers',   # >60% A+G
        'repeat_kmers'         # All same nucleotide
    ]
    
    if not clipped_sequences:
        return {name: 0 for name in feature_names}
    
    composition_counts = {name: 0 for name in feature_names}
    
    for seq in clipped_sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(base in 'ATGC' for base in kmer):  # Valid nucleotides only
                gc_content = (kmer.count('G') + kmer.count('C')) / k
                purine_content = (kmer.count('A') + kmer.count('G')) / k
                
                if len(set(kmer)) == 1:  # All same nucleotide
                    composition_counts['repeat_kmers'] += 1
                elif gc_content > 0.6:
                    composition_counts['gc_rich_kmers'] += 1
                elif gc_content < 0.4:
                    composition_counts['at_rich_kmers'] += 1
                else:
                    composition_counts['balanced_kmers'] += 1
                    
                if purine_content > 0.6:
                    composition_counts['purine_rich_kmers'] += 1
    
    # Return in consistent order
    return {name: composition_counts[name] for name in feature_names}

def softclip_kmer_features(clipped_sequences, k=3):
    """Hybrid approach: combine fixed k-mers with statistical features."""
    fixed_features = softclip_fixed_kmer_features(clipped_sequences, k)
    stats_features = softclip_kmer_stats(clipped_sequences, k) 
    composition_features = softclip_kmer_composition(clipped_sequences, k)
    
    # Combine all features in consistent order
    combined = {}
    # 1. First add fixed k-mers (already sorted)
    combined.update(fixed_features)
    # 2. Then add statistical features (already ordered)
    combined.update(stats_features)
    # 3. Finally add composition features (already ordered)
    combined.update(composition_features)
    
    return combined

def softclip_composition_features(clipped_sequences, prefix=""):
    """Calculate nucleotide composition features."""
    # Define feature names in consistent order
    feature_names = [
        f'{prefix}at_content',
        f'{prefix}gc_content', 
        f'{prefix}purine_ratio'
    ]
    
    if not clipped_sequences:
        return {name: 0 for name in feature_names}
    
    all_bases = ''.join(clipped_sequences)
    total = len(all_bases)
    if total == 0:
        return {name: 0 for name in feature_names}
    
    # Return in consistent order
    return {
        f'{prefix}at_content': (all_bases.count('A') + all_bases.count('T')) / total,
        f'{prefix}gc_content': (all_bases.count('G') + all_bases.count('C')) / total,
        f'{prefix}purine_ratio': (all_bases.count('A') + all_bases.count('G')) / total
    }

def softclip_motif_features(clipped_sequences):
    """Detect biologically relevant motifs in soft-clipped sequences."""
    # Define feature names in consistent order
    feature_names = [
        'cg_rich',
        'polyA_signals',
        'splice_signals',
        'tata_like'
    ]
    
    if not clipped_sequences:
        return {name: 0 for name in feature_names}
    
    # Count relevant motifs
    polyA_count = len([seq for seq in clipped_sequences if 'AAAA' in seq.upper()])
    tata_count = len([seq for seq in clipped_sequences if 'TATA' in seq.upper()])
    splice_count = len([seq for seq in clipped_sequences if 'GT' in seq.upper() or 'AG' in seq.upper()])
    cg_rich_count = len([seq for seq in clipped_sequences if seq.upper().count('CG') >= 2])
    
    # Return in consistent order
    return {
        'cg_rich': cg_rich_count,
        'polyA_signals': polyA_count,
        'splice_signals': splice_count,
        'tata_like': tata_count
    }

def softclip_homopolymer_features(clipped_sequences, prefix=""):
    """Detect homopolymer runs in soft-clipped sequences."""
    # Define feature names in consistent order
    bases = ['A', 'C', 'G', 'T']  # Alphabetical order
    
    if not clipped_sequences:
        return {f'{prefix}max_poly{base}': 0 for base in bases}
    
    runs = {base: 0 for base in bases}
    
    for seq in clipped_sequences:
        seq = seq.upper()
        for base in bases:
            matches = re.findall(f'{base}+', seq)
            if matches:
                max_run = max(len(match) for match in matches)
                runs[base] = max(runs[base], max_run)
    
    # Return in consistent order
    return {f'{prefix}max_poly{base}': runs[base] for base in bases}


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
    embedding_features = extract_kmer_embeddings(start_clip_sequences, end_clip_sequences, cfg, cfg.current_site_type)
    
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
    
    # Collect features
    # if os.path.exists(cfg.tss_feature_file) and os.path.exists(cfg.tes_feature_file):
    #     print(f"Feature files already exist: {cfg.tss_feature_file}, {cfg.tes_feature_file}")
    #     # Close the BAM file
    #     bam.close()
    #     save_config(config_path)

    #     return

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
        embedding_type = getattr(cfg, 'embedding_type', 'cnn')
        if embedding_type == 'kmer':
            print("Cleaning up k-mer embedders...")
            cleanup_kmer_embedders()
        # elif embedding_type == 'cnn':
        #     print("Cleaning up CNN embedders...")
        #     from sequence_cnn_embeddings import cleanup_cnn_embedder
        #     cleanup_cnn_embedder()
        # else:
        #     print("Cleaning up DNABERT embedder...")
        #     cleanup_embedder()
        # 
    save_config(config_path)
    
    print("Feature extraction complete! Output saved as 'candidate_site_features.csv'.")


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