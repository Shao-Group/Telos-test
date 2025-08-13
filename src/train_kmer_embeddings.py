#!/usr/bin/env python3
"""
Train XGBoost/Random Forest models using k-mer embeddings instead of CNN.
This should work much better for soft-clipped sequences.
"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

from config import load_config, save_config
from kmer_embeddings import EnhancedSequenceEmbedder


def collect_site_data_for_kmer_embedding(softclip_file, clip_type, site_type, max_sites_per_type=5000):
    """
    Collect site data for k-mer embedding (same as CNN but simpler).
    """
    print(f"Loading {site_type} {clip_type} soft-clipped sequences for k-mer embedding...")
    
    if not os.path.exists(softclip_file):
        print(f"File not found: {softclip_file}")
        return []
    
    df = pd.read_csv(softclip_file)
    df_filtered = df[df['clip_type'] == clip_type]
    
    # Use more chromosomes for k-mer approach since it's more robust
    train_chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
    chrom_mask = df_filtered['chrom'].astype(str).str.startswith('chr')
    df_filtered = df_filtered[chrom_mask]
    df_filtered = df_filtered[df_filtered['chrom'].isin(train_chroms)]
    
    print(f"Found {len(df_filtered)} {site_type} {clip_type} clips")
    
    # Group by site
    site_grouped = defaultdict(lambda: {'sequences': [], 'labels': []})
    
    for _, row in df_filtered.iterrows():
        site_key = (row['chrom'], row['position'], row['strand'])
        seq = str(row['sequence']).strip()
        
        # More lenient filtering for k-mers
        if seq and seq != 'nan' and len(seq) >= 5:  # Minimum 5bp
            site_grouped[site_key]['sequences'].append(seq)
            site_grouped[site_key]['labels'].append(row['label'])
    
    # Convert to site data
    site_data = []
    for site_key, site_info in site_grouped.items():
        if len(site_info['sequences']) >= 1:  # At least 1 sequence
            # Handle label consistency
            unique_labels = set(site_info['labels'])
            if len(unique_labels) > 1:
                site_label = max(set(site_info['labels']), key=site_info['labels'].count)
            else:
                site_label = site_info['labels'][0]
            
            site_data.append({
                'sequences': site_info['sequences'],
                'label': site_label,
                'site_type': site_type,
                'clip_type': clip_type,
                'position': f"{site_key[0]}:{site_key[1]}:{site_key[2]}"
            })
    
    print(f"Collected {len(site_data)} sites")
    
    # Balance the data
    import random
    pos_sites = [site for site in site_data if site['label'] == 1]
    neg_sites = [site for site in site_data if site['label'] == 0]
    
    print(f"Before balancing: {len(pos_sites)} positive, {len(neg_sites)} negative")
    
    if len(pos_sites) > 0 and len(neg_sites) > 0:
        n_samples = min(len(pos_sites), len(neg_sites), max_sites_per_type // 2)
        pos_sample = random.sample(pos_sites, min(n_samples, len(pos_sites)))
        neg_sample = random.sample(neg_sites, min(n_samples, len(neg_sites)))
        
        site_data = pos_sample + neg_sample
        random.shuffle(site_data)
        
        print(f"After balancing: {len(pos_sample)} positive, {len(neg_sample)} negative")
    
    return site_data


def train_kmer_model(site_data, clip_type, site_type, model_type='xgboost'):
    """Train model using k-mer embeddings."""
    print(f"\n🚀 Training {model_type} model for {clip_type} {site_type}")
    print("-" * 50)
    
    if len(site_data) < 10:
        print(f"Not enough data: {len(site_data)} sites")
        return None, None
    
    # Initialize embedder
    embedder = EnhancedSequenceEmbedder(
        k_sizes=[3, 4, 5],
        clip_type=f"{clip_type}_{site_type}"
    )
    
    # Generate embeddings
    print("Generating k-mer embeddings...")
    X = []
    y = []
    
    for site in site_data:
        embedding = embedder.embed_site(site['sequences'])
        X.append(embedding)
        y.append(site['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    if model_type.lower() == 'xgboost':
        print("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_names = embedder.get_feature_names()
        importances = model.feature_importances_
        
        # Top 10 features
        top_indices = np.argsort(importances)[-10:][::-1]
        print("\nTop 10 most important features:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. {feature_names[idx]:<30} {importances[idx]:.4f}")
    
    return model, embedder


def train_all_kmer_models(config_path, model_type='xgboost', max_sites_per_type=5000):
    """Train k-mer embedding models for all clip types and site types."""
    
    cfg = load_config(config_path)
    
    # Check files
    tss_softclip_file = getattr(cfg, 'tss_softclip_labeled_file', None)
    tes_softclip_file = getattr(cfg, 'tes_softclip_labeled_file', None)
    
    if not tss_softclip_file or not tes_softclip_file:
        print("Error: Soft-clip labeled files not found!")
        return {}
    
    print("🧬 Training K-mer Embedding Models")
    print("=" * 50)
    print(f"Model type: {model_type}")
    print(f"Max sites per type: {max_sites_per_type}")
    
    models = {}
    embedders = {}
    
    # Train all 4 combinations
    for clip_type in ['start', 'end']:
        for site_type, softclip_file in [('TSS', tss_softclip_file), ('TES', tes_softclip_file)]:
            model_key = f"{clip_type}_{site_type}"
            
            # Collect data
            site_data = collect_site_data_for_kmer_embedding(
                softclip_file, clip_type, site_type, max_sites_per_type
            )
            
            if len(site_data) >= 10:
                # Train model
                model, embedder = train_kmer_model(
                    site_data, clip_type, site_type, model_type
                )
                
                if model is not None:
                    models[model_key] = model
                    embedders[model_key] = embedder
                    
                    # Save model
                    model_filename = f"kmer_{model_type}_{clip_type}_{site_type}_model.pkl"
                    embedder_filename = f"kmer_embedder_{clip_type}_{site_type}.pkl"
                    
                    model_path = os.path.join(cfg.models_output_dir, model_filename)
                    embedder_path = os.path.join(cfg.models_output_dir, embedder_filename)
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    with open(embedder_path, 'wb') as f:
                        pickle.dump(embedder, f)
                    
                    print(f"✅ Saved {model_key} model to {model_path}")
            else:
                print(f"❌ Skipping {model_key}: insufficient data ({len(site_data)} sites)")
    
    print(f"\n🎉 Training completed! Trained {len(models)} models.")
    return models, embedders


def main():
    parser = argparse.ArgumentParser(description="Train k-mer embedding models")
    parser.add_argument("--config", required=True, help="Configuration file")
    parser.add_argument("--model-type", choices=['xgboost', 'randomforest'], 
                       default='xgboost', help="Model type to train")
    parser.add_argument("--max-sites-per-type", type=int, default=5000, 
                       help="Max sites per type")
    
    args = parser.parse_args()
    
    models, embedders = train_all_kmer_models(
        config_path=args.config,
        model_type=args.model_type,
        max_sites_per_type=args.max_sites_per_type
    )
    
    if len(models) > 0:
        print(f"\n✅ Successfully trained {len(models)} k-mer models!")
        print("These should perform MUCH better than the CNN approach.")
    else:
        print("\n❌ No models were trained successfully.")


if __name__ == "__main__":
    main()