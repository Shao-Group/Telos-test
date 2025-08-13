#!/usr/bin/env python3
"""
Ultra-simple training script for site-level CNN.
Stripped down to bare minimum.
"""

import argparse
import pandas as pd
import os
import sys
import gc
import torch
from collections import defaultdict
from config import load_config, save_config
from site_cnn_embeddings import SiteCNNTrainer

def collect_site_data_by_clip_and_site_type(softclip_file, clip_type, site_type, max_sites_per_type=5000):
    """
    Collect candidate site data for specific clip type and site type (TSS or TES).
    """
    site_data = []
    
    print(f"Loading {site_type} {clip_type} soft-clipped sequences...")
    
    if not os.path.exists(softclip_file):
        print(f"File not found: {softclip_file}")
        return site_data
    
    # Load data
    print(f"Loading {site_type} {clip_type} clips from {softclip_file}")
    df = pd.read_csv(softclip_file)
    
    # Filter by clip type and training chromosomes
    df_filtered = df[df['clip_type'] == clip_type]
    train_chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
    chrom_mask = df_filtered['chrom'].astype(str).str.startswith('chr')
    df_filtered = df_filtered[chrom_mask]
    df_filtered = df_filtered[df_filtered['chrom'].isin(train_chroms)]
    
    print(f"Found {len(df_filtered)} {site_type} {clip_type} clips")
    
    # Group sequences by candidate site
    site_grouped = defaultdict(lambda: {'sequences': [], 'labels': []})
    
    for _, row in df_filtered.iterrows():
        site_key = (row['chrom'], row['position'], row['strand'])
        seq = str(row['sequence'])
        
        if seq and seq != 'nan' and len(seq) >= 2:
            site_grouped[site_key]['sequences'].append(seq)
            site_grouped[site_key]['labels'].append(row['label'])
    
    # Convert to site-level data
    for site_key, site_info in site_grouped.items():
        if len(site_info['sequences']) > 0:
            # Check label consistency - all sequences at a site should have same label
            unique_labels = set(site_info['labels'])
            if len(unique_labels) > 1:
                print(f"Warning: Mixed labels at site {site_key}: {unique_labels}")
                # Use majority vote
                site_label = max(set(site_info['labels']), key=site_info['labels'].count)
            else:
                site_label = site_info['labels'][0]
            
            # Only include sites with enough sequences
            if len(site_info['sequences']) >= 2:  # At least 2 sequences per site
                site_data.append({
                    'sequences': site_info['sequences'],
                    'label': site_label,
                    'site_type': site_type,
                    'clip_type': clip_type,
                    'position': f"{site_key[0]}:{site_key[1]}:{site_key[2]}"
                })
    
    print(f"Loaded {len(site_grouped)} {site_type} candidate sites")
    
    # Statistics
    total_sites = len(site_data)
    positive_sites = sum(1 for site in site_data if site['label'] == 1)
    negative_sites = total_sites - positive_sites
    
    print(f"Total {site_type} {clip_type} candidate sites: {total_sites}")
    print(f"Positive sites: {positive_sites}, Negative sites: {negative_sites}")
    
    # Better balancing - ensure equal positive and negative samples
    import random
    pos_sites = [site for site in site_data if site['label'] == 1]
    neg_sites = [site for site in site_data if site['label'] == 0]
    
    print(f"Before balancing: {len(pos_sites)} positive, {len(neg_sites)} negative")
    
    if len(pos_sites) > 0 and len(neg_sites) > 0:
        # Take equal amounts of each class, up to max_sites_per_type
        n_samples = min(len(pos_sites), len(neg_sites), max_sites_per_type // 2)  # Divide by 2 since we take both pos and neg
        
        pos_sample = random.sample(pos_sites, n_samples)
        neg_sample = random.sample(neg_sites, n_samples)
        
        site_data = pos_sample + neg_sample
        random.shuffle(site_data)  # Shuffle to mix positive and negative
        
        print(f"After balancing: {n_samples} positive, {n_samples} negative")
        print(f"Total balanced sites: {len(site_data)}")
    else:
        print(f"Warning: Imbalanced data - pos: {len(pos_sites)}, neg: {len(neg_sites)}")
    
    return site_data

def train_site_cnn(cfg, clip_type, site_type, site_data, epochs, batch_size, max_seq_length, embedding_dim):
    """Train a simple site CNN model."""
    if len(site_data) == 0:
        print(f"No {clip_type} {site_type} site data found!")
        return None
    
    print(f"Training {clip_type} {site_type} site CNN:")
    print(f"  - Sites: {len(site_data)}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - Max seq length: {max_seq_length}")
    
    # Initialize trainer
    trainer = SiteCNNTrainer(
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        clip_type=f"{clip_type}_{site_type}"
    )
    
    # Train
    print(f"Training for {epochs} epochs...")
    history = trainer.train(
        site_data=site_data,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Save model
    model_filename = f"simple_site_cnn_{clip_type}_{site_type}_model.pth"
    model_path = os.path.join(cfg.models_output_dir, model_filename)
    trainer.save_model(model_path)
    
    print(f"{clip_type.capitalize()} {site_type} CNN training complete! Model saved to {model_path}")
    
    # Print summary
    if history['val_acc']:
        best_val_acc = max(history['val_acc'])
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model_path

def train_site_cnn_models(config_path, epochs=50, batch_size=16, max_seq_length=50, 
                          embedding_dim=32, max_sites_per_type=2000):
    """Train separate models for each clip type and site type (4 models total)."""
    
    # GPU setup
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load config
    cfg = load_config(config_path)
    cfg.set_embedding_type('site_cnn')
    cfg.set_embedding_max_length(max_seq_length)
    cfg.set_embedding_dim(embedding_dim)
    
    # Check files
    tss_softclip_file = getattr(cfg, 'tss_softclip_labeled_file', None)
    tes_softclip_file = getattr(cfg, 'tes_softclip_labeled_file', None)
    
    if not tss_softclip_file or not tes_softclip_file:
        print("Error: Soft-clip labeled files not found!")
        return {}
    
    if not os.path.exists(tss_softclip_file) or not os.path.exists(tes_softclip_file):
        print("Error: Soft-clip files do not exist!")
        return {}
    
    print("🧬 Training Separate Site-Level CNN Models")
    print("=" * 50)
    print("Training 4 models: start_TSS, start_TES, end_TSS, end_TES")
    
    model_paths = {}
    
    # Train all 4 combinations
    for clip_type in ['start', 'end']:
        for site_type, softclip_file in [('TSS', tss_softclip_file), ('TES', tes_softclip_file)]:
            print(f"\n🚀 Training {clip_type} {site_type} model...")
            print("-" * 40)
            
            # Collect data for this specific combination
            site_data = collect_site_data_by_clip_and_site_type(
                softclip_file, clip_type, site_type, max_sites_per_type
            )
            
            if len(site_data) > 0:
                # Train model
                model_path = train_site_cnn(
                    cfg, clip_type, site_type, site_data, epochs, batch_size,
                    max_seq_length, embedding_dim
                )
                
                if model_path:
                    model_paths[f"{clip_type}_{site_type}"] = model_path
            else:
                print(f"No data found for {clip_type} {site_type}, skipping...")
            
            # Clean up
            del site_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Update config with all model paths
    if 'start_TSS' in model_paths:
        cfg.set_cnn_start_tss_model_path(model_paths['start_TSS'])
    if 'start_TES' in model_paths:
        cfg.set_cnn_start_tes_model_path(model_paths['start_TES'])
    if 'end_TSS' in model_paths:
        cfg.set_cnn_end_tss_model_path(model_paths['end_TSS'])
    if 'end_TES' in model_paths:
        cfg.set_cnn_end_tes_model_path(model_paths['end_TES'])
    
    save_config(config_path)
    
    print("\n✅ Site CNN training completed!")
    print("Trained models:")
    for model_name, model_path in model_paths.items():
        print(f"   {model_name}: {model_path}")
    
    return model_paths

def main():
    parser = argparse.ArgumentParser(description="Train simple site-level CNN")
    parser.add_argument("--config", required=True, help="Configuration file")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-seq-length", type=int, default=50, help="Max sequence length")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--max-sites-per-type", type=int, default=2000, help="Max sites per type")
    
    args = parser.parse_args()
    
    print("🧬 Training Simple Site-Level CNN")
    print("=" * 40)
    
    model_paths = train_site_cnn_models(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        max_sites_per_type=args.max_sites_per_type
    )
    
    if len(model_paths) > 0:
        print(f"\n🎉 Training successful! Trained {len(model_paths)} models.")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()