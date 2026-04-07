#!/usr/bin/env python3
"""
K-mer based embeddings for soft-clipped sequences.
Much more suitable for short sequences than CNNs.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from itertools import product
import pickle
import os


class KmerEmbedder:
    """K-mer based sequence embedder for soft clips."""
    
    def __init__(self, k_sizes=[3, 4, 5], include_reverse_complement=True, 
                 normalize=True, clip_type='start'):
        """
        Initialize k-mer embedder.
        
        Args:
            k_sizes: List of k-mer sizes to use
            include_reverse_complement: Include reverse complement k-mers
            normalize: Normalize k-mer counts to frequencies
            clip_type: Type of clipping (start/end)
        """
        self.k_sizes = k_sizes
        self.include_reverse_complement = include_reverse_complement
        self.normalize = normalize
        self.clip_type = clip_type
        
        # Generate all possible k-mers for each k
        self.kmer_to_idx = {}
        self.feature_names = []
        
        self._build_kmer_vocabulary()
        
    def _build_kmer_vocabulary(self):
        """Build vocabulary of all possible k-mers using shared vocabulary."""
        # Use shared vocabulary to ensure consistent feature dimensions
        self.kmer_to_idx, self.feature_names, self.n_features = _build_shared_kmer_vocabulary(
            self.k_sizes, self.include_reverse_complement
        )
        print(f"Built k-mer vocabulary: {self.n_features} features")
    
    def _reverse_complement(self, seq):
        """Get reverse complement of DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))
    
    def _extract_kmers(self, sequence):
        """Extract k-mers from a sequence."""
        sequence = sequence.upper().replace('N', '')  # Remove N's
        if len(sequence) < min(self.k_sizes):
            return Counter()
        
        kmer_counts = Counter()
        
        for k in self.k_sizes:
            if len(sequence) >= k:
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i:i+k]
                    if 'N' not in kmer:  # Skip k-mers with N
                        kmer_counts[kmer] += 1
                        
                        # Add reverse complement
                        if self.include_reverse_complement:
                            rev_kmer = self._reverse_complement(kmer)
                            if rev_kmer != kmer:
                                kmer_counts[rev_kmer] += 1
        
        return kmer_counts
    
    def _sequence_to_vector(self, sequence):
        """Convert single sequence to k-mer feature vector."""
        kmer_counts = self._extract_kmers(sequence)
        
        # Create feature vector
        vector = np.zeros(self.n_features)
        
        for kmer, count in kmer_counts.items():
            if kmer in self.kmer_to_idx:
                vector[self.kmer_to_idx[kmer]] = count
        
        # Normalize to frequencies if requested
        if self.normalize and vector.sum() > 0:
            vector = vector / vector.sum()
        
        return vector
    
    def embed_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Convert list of sequences to k-mer feature matrix.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Feature matrix of shape (n_sequences, n_features)
        """
        if not sequences:
            return np.zeros((0, self.n_features))
        
        vectors = []
        for seq in sequences:
            if seq and seq.strip() and seq != 'nan':
                vector = self._sequence_to_vector(seq.strip())
                vectors.append(vector)
        
        if not vectors:
            return np.zeros((0, self.n_features))
        
        return np.array(vectors)
    
    def embed_site(self, site_sequences: List[str]) -> np.ndarray:
        """
        Create site-level embedding from multiple sequences.
        
        Args:
            site_sequences: List of sequences from the same genomic site
            
        Returns:
            Site embedding vector
        """
        seq_vectors = self.embed_sequences(site_sequences)
        
        if seq_vectors.shape[0] == 0:
            # Return zeros for mean + max aggregated features
            return np.zeros(2 * self.n_features)
        
        # Aggregate sequences from the same site
        # Use mean + max only (remove std to save 1/3 memory)
        site_embedding = np.concatenate([
            seq_vectors.mean(axis=0),  # Average k-mer profile
            seq_vectors.max(axis=0),   # Max k-mer counts
        ])
        
        return site_embedding
    
    def get_feature_names(self):
        """Get names of all features."""
        base_names = self.feature_names.copy()
        
        # Add aggregation prefixes for site-level features
        feature_names = []
        for prefix in ['mean', 'max']:  # Removed 'std' to save memory
            for name in base_names:
                feature_names.append(f"{prefix}_{name}")
        
        return feature_names


class EnhancedSequenceEmbedder:
    """Enhanced embedder with k-mers + statistical features."""
    
    def __init__(self, k_sizes=[3, 4,5], clip_type='start'):  
        self.clip_type = clip_type
        # Ensure all embedders use the same parameters for consistent vocabulary
        self.kmer_embedder = KmerEmbedder(
            k_sizes=k_sizes, 
            include_reverse_complement=True, 
            normalize=True, 
            clip_type=clip_type
        )
        
    def _calculate_statistical_features(self, sequences: List[str]) -> np.ndarray:
        """Calculate statistical features for sequences."""
        if not sequences:
            return np.zeros(13)  # 13 statistical features
        
        valid_sequences = [seq.strip().upper() for seq in sequences 
                          if seq and seq.strip() and seq != 'nan']
        
        if not valid_sequences:
            return np.zeros(13)
        
        features = []
        
        # Length statistics
        lengths = [len(seq) for seq in valid_sequences]
        features.extend([
            np.mean(lengths),
            np.std(lengths) if len(lengths) > 1 else 0,
            np.min(lengths),
            np.max(lengths)
        ])
        
        # Nucleotide composition
        all_seq = ''.join(valid_sequences)
        total_bases = len(all_seq)
        
        if total_bases > 0:
            gc_content = (all_seq.count('G') + all_seq.count('C')) / total_bases
            at_content = (all_seq.count('A') + all_seq.count('T')) / total_bases
            n_content = all_seq.count('N') / total_bases
            
            features.extend([gc_content, at_content, n_content])
        else:
            features.extend([0, 0, 0])
        
        # Sequence complexity (entropy)
        if total_bases > 0:
            base_counts = Counter(all_seq)
            entropy = -sum((count/total_bases) * np.log2(count/total_bases + 1e-10) 
                          for count in base_counts.values())
            features.append(entropy)
        else:
            features.append(0)
        
        # Number of sequences at this site
        features.append(len(valid_sequences))
        
        # Dinucleotide features
        dinucs = ['AA', 'AT', 'GC', 'CG']
        for dinuc in dinucs:
            count = sum(seq.count(dinuc) for seq in valid_sequences)
            features.append(count / max(total_bases - len(valid_sequences), 1))
        
        return np.array(features)
    
    def embed_site(self, site_sequences: List[str]) -> np.ndarray:
        """Create comprehensive site embedding."""
        # K-mer features
        kmer_features = self.kmer_embedder.embed_site(site_sequences)
        
        # Statistical features
        stat_features = self._calculate_statistical_features(site_sequences)
        
        # Combine all features
        return np.concatenate([kmer_features, stat_features])
    
    def get_feature_names(self):
        """Get all feature names."""
        kmer_names = self.kmer_embedder.get_feature_names()
        
        stat_names = [
            'mean_length', 'std_length', 'min_length', 'max_length',
            'gc_content', 'at_content', 'n_content', 'entropy',
            'num_sequences',
            'dinuc_AA_freq', 'dinuc_AT_freq', 'dinuc_GC_freq', 'dinuc_CG_freq'
        ]
        
        return kmer_names + stat_names


# Global embedder instances
_global_kmer_start_tss_embedder = None
_global_kmer_start_tes_embedder = None
_global_kmer_end_tss_embedder = None
_global_kmer_end_tes_embedder = None

# Shared vocabulary for consistent feature dimensions
_shared_kmer_vocabulary = None


def _build_shared_kmer_vocabulary(k_sizes=[3, 4, 5], include_reverse_complement=True):
    """Build shared k-mer vocabulary for all embedders."""
    global _shared_kmer_vocabulary
    
    if _shared_kmer_vocabulary is not None:
        return _shared_kmer_vocabulary
    
    bases = ['A', 'T', 'G', 'C']
    kmer_to_idx = {}
    feature_names = []
    idx = 0
    
    for k in sorted(k_sizes):  # Sort to ensure consistent order
        # Generate all possible k-mers
        kmers = [''.join(kmer) for kmer in product(bases, repeat=k)]
        kmers.sort()  # Sort for consistency
        
        for kmer in kmers:
            if kmer not in kmer_to_idx:
                kmer_to_idx[kmer] = idx
                feature_names.append(f"{k}mer_{kmer}")
                idx += 1
            
            # Add reverse complement if requested
            if include_reverse_complement:
                complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
                rev_comp = ''.join(complement.get(base, 'N') for base in reversed(kmer))
                if rev_comp not in kmer_to_idx and rev_comp != kmer:
                    kmer_to_idx[rev_comp] = idx
                    feature_names.append(f"{k}mer_{rev_comp}")
                    idx += 1
    
    _shared_kmer_vocabulary = (kmer_to_idx, feature_names, len(feature_names))
    return _shared_kmer_vocabulary


def get_kmer_start_tss_embedder() -> EnhancedSequenceEmbedder:
    """Get global start TSS k-mer embedder."""
    global _global_kmer_start_tss_embedder
    
    if _global_kmer_start_tss_embedder is None:
        _global_kmer_start_tss_embedder = EnhancedSequenceEmbedder(clip_type='start_TSS')
    return _global_kmer_start_tss_embedder


def get_kmer_start_tes_embedder() -> EnhancedSequenceEmbedder:
    """Get global start TES k-mer embedder."""
    global _global_kmer_start_tes_embedder
    
    if _global_kmer_start_tes_embedder is None:
        _global_kmer_start_tes_embedder = EnhancedSequenceEmbedder(clip_type='start_TES')
    return _global_kmer_start_tes_embedder


def get_kmer_end_tss_embedder() -> EnhancedSequenceEmbedder:
    """Get global end TSS k-mer embedder."""
    global _global_kmer_end_tss_embedder
    
    if _global_kmer_end_tss_embedder is None:
        _global_kmer_end_tss_embedder = EnhancedSequenceEmbedder(clip_type='end_TSS')
    return _global_kmer_end_tss_embedder


def get_kmer_end_tes_embedder() -> EnhancedSequenceEmbedder:
    """Get global end TES k-mer embedder."""
    global _global_kmer_end_tes_embedder
    
    if _global_kmer_end_tes_embedder is None:
        _global_kmer_end_tes_embedder = EnhancedSequenceEmbedder(clip_type='end_TES')
    return _global_kmer_end_tes_embedder


def get_separate_kmer_embedders():
    """Get all 4 k-mer embedders."""
    start_tss = get_kmer_start_tss_embedder()
    start_tes = get_kmer_start_tes_embedder()
    end_tss = get_kmer_end_tss_embedder()
    end_tes = get_kmer_end_tes_embedder()
    return start_tss, start_tes, end_tss, end_tes


def cleanup_kmer_embedders():
    """Clean up all global embedders."""
    global _global_kmer_start_tss_embedder, _global_kmer_start_tes_embedder
    global _global_kmer_end_tss_embedder, _global_kmer_end_tes_embedder
    global _shared_kmer_vocabulary
    
    _global_kmer_start_tss_embedder = None
    _global_kmer_start_tes_embedder = None
    _global_kmer_end_tss_embedder = None
    _global_kmer_end_tes_embedder = None
    _shared_kmer_vocabulary = None