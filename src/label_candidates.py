import numpy as np
import argparse
import pandas as pd
import os
from tqdm import tqdm
from ml_utils import normalize_chrom
from config import load_config
import sys

def load_candidates(candidate_path):
    """Load extracted candidate features."""
    return pd.read_csv(candidate_path, dtype={"chrom": str, "position": int, "strand": str})

def load_reference(reference_path):
    """Load reference annotation TSS/TES positions."""
    ref_df = pd.read_csv(reference_path, sep=" ", header=None,
                         names=["site_type", "chrom", "position", "pos_strand_cnt", "neg_strand_cnt"],
                         dtype={"chrom": str})
    ref_df["strand"] = ref_df.apply(
        lambda x: "+" if x["pos_strand_cnt"] > x["neg_strand_cnt"] else "-", axis=1
    )
    ref_df = ref_df[["site_type", "chrom", "position", "strand"]]
    # print(ref_df.head(5))
    return ref_df



def label_candidates(candidate_df, reference_df, site_type, max_distance=50):
    """Fast labeling using grouped and vectorized search."""
    
    reference_df = reference_df[
        reference_df['site_type'].str.lower() == site_type.lower()
    ].copy()
    reference_df['chrom'] = reference_df['chrom'].astype(str)
    reference_df['position'] = reference_df['position'].astype(int)

    # Group reference sites for fast lookup
    grouped_ref = reference_df.groupby(['chrom', 'strand'])

    labels = np.zeros(len(candidate_df), dtype=int)

    print(f"Labeling {site_type.upper()} candidates with {max_distance} nt tolerance ...")
    for i, row in candidate_df.iterrows():
        chrom, pos, strand = row['chrom'], row['position'], row['strand']
        if i % 1000 == 0:
            print(f"Labeled {i} candidates")
            sys.stdout.flush()
            
        try:
            group = grouped_ref.get_group((chrom, strand))
        except KeyError:
            continue  # No matching chrom+strand in reference
        
        # Use numpy for fast range check
        ref_positions = group['position'].values
        match_found = np.any(np.abs(ref_positions - pos) <= max_distance)
        labels[i] = 1 if match_found else 0

    candidate_df['label'] = labels
    return candidate_df


def label_softclip_sequences(softclip_df, reference_df, site_type, max_distance=50):
    """
    Label individual soft-clipped sequences based on their candidate site positions.
    
    Args:
        softclip_df: DataFrame with soft-clipped sequences
        reference_df: DataFrame with reference annotations
        site_type: 'TSS' or 'TES'
        max_distance: Maximum distance for matching
        
    Returns:
        DataFrame with labels added
    """
    reference_df = reference_df[
        reference_df['site_type'].str.lower() == site_type.lower()
    ].copy()
    reference_df['chrom'] = reference_df['chrom'].astype(str)
    reference_df['position'] = reference_df['position'].astype(int)

    # Group reference sites for fast lookup
    grouped_ref = reference_df.groupby(['chrom', 'strand'])
    grouped_softclip_df = softclip_df.groupby(['chrom', 'strand', 'position'])

    labels = np.zeros(len(softclip_df), dtype=int)
    softclip_df['label'] = labels

    print(f"Labeling {site_type.upper()} soft-clipped sequences with {max_distance} nt tolerance ...")
    i = 0
    for key, row in grouped_softclip_df:
        if i % 1000 == 0:
            print(f"Labeled {i} soft-clipped sequences")
            sys.stdout.flush()
        i += 1

        chrom, strand, pos = key
        idx = grouped_softclip_df.indices[key]
        
        try:
            group = grouped_ref.get_group((chrom, strand))
        except KeyError:
            continue  # No matching chrom+strand in reference

        # Use numpy for fast range check
        ref_positions = group['position'].values
        match_found = np.any(np.abs(ref_positions - pos) <= max_distance)
        softclip_df.loc[idx, 'label'] = 1 if match_found else 0

    # softclip_df['label'] = labels
    return softclip_df


def main(args, cfg):
    # Load configuration
    reference_df = load_reference(cfg.ref_candidate_file)
    
    # Label feature candidates if they exist
    if os.path.exists(cfg.tss_feature_file) and os.path.exists(cfg.tes_feature_file):
        # Load candidates
        tss_candidate_df = load_candidates(cfg.tss_feature_file)
        tes_candidate_df = load_candidates(cfg.tes_feature_file)

        # Label candidates
        tss_labeled_df = label_candidates(tss_candidate_df, reference_df, "TSS", args.distance)
        tes_labeled_df = label_candidates(tes_candidate_df, reference_df, "TES", args.distance)

        # Save labeled candidates
        tss_labeled_df.to_csv(cfg.tss_labeled_file, index=False)
        tes_labeled_df.to_csv(cfg.tes_labeled_file, index=False)
        print(f"Labeled TSS candidates saved to: {cfg.tss_labeled_file}")
        print(f"Labeled TES candidates saved to: {cfg.tes_labeled_file}")
    
    # Label soft-clipped sequences if they exist
    if hasattr(cfg, 'tss_softclip_file') and hasattr(cfg, 'tes_softclip_file'):
        if os.path.exists(cfg.tss_softclip_file) and os.path.exists(cfg.tes_softclip_file):
            # Load soft-clipped sequences
            tss_softclip_df = pd.read_csv(cfg.tss_softclip_file, dtype={"chrom": str})
            tes_softclip_df = pd.read_csv(cfg.tes_softclip_file, dtype={"chrom": str})
            
            # Label soft-clipped sequences
            tss_softclip_labeled_df = label_softclip_sequences(tss_softclip_df, reference_df, "TSS", args.distance)
            tes_softclip_labeled_df = label_softclip_sequences(tes_softclip_df, reference_df, "TES", args.distance)
            
            # Save labeled soft-clipped sequences
            tss_softclip_labeled_df.to_csv(cfg.tss_softclip_labeled_file, index=False)
            tes_softclip_labeled_df.to_csv(cfg.tes_softclip_labeled_file, index=False)
            print(f"Labeled TSS soft-clipped sequences saved to: {cfg.tss_softclip_labeled_file}")
            print(f"Labeled TES soft-clipped sequences saved to: {cfg.tes_softclip_labeled_file}")
            
            # Print summary statistics
            print(f"\nSoft-clipped sequence labeling summary:")
            print(f"TSS sequences: {len(tss_softclip_labeled_df)} total, {sum(tss_softclip_labeled_df['label'])} positive")
            print(f"TES sequences: {len(tes_softclip_labeled_df)} total, {sum(tes_softclip_labeled_df['label'])} positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label candidate TSS/TES sites based on reference.")
    parser.add_argument('-d', '--distance', type=int, default=50, help="Maximum distance allowed for matching.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(args, cfg)
