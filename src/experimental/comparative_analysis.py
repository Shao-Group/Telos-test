#!/usr/bin/env python3
"""
Comparative Analysis Script: cDNA vs dRNA Performance

This script performs comprehensive comparative analysis to explain why dRNA 
performs worse than cDNA by analyzing:
1. Feature distribution differences
2. Signal-to-noise ratio variations
3. 5' degradation impact analysis

Focus: Understanding dRNA limitations compared to cDNA
"""

import argparse
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, classification_report
)
import pysam
from collections import Counter, defaultdict
from config import load_config
from ml_utils import load_saved_model
from label_candidates import load_selected_features
import warnings
warnings.filterwarnings('ignore')

class ComparativeAnalyzer:
    """Main class for performing cDNA vs dRNA comparative analysis."""
    
    def __init__(self, cdna_config, drna_config):
        self.cdna_config = cdna_config
        self.drna_config = drna_config
        # Use predictions directories from config
        self.cdna_predictions_dir = cdna_config.predictions_output_dir
        self.drna_predictions_dir = drna_config.predictions_output_dir
        
        # Load datasets
        print("Loading cDNA datasets...")
        self.cdna_tss = pd.read_csv(cdna_config.tss_labeled_file, dtype={"chrom": str})
        self.cdna_tes = pd.read_csv(cdna_config.tes_labeled_file, dtype={"chrom": str})
        
        print("Loading dRNA datasets...")
        self.drna_tss = pd.read_csv(drna_config.tss_labeled_file, dtype={"chrom": str})
        self.drna_tes = pd.read_csv(drna_config.tes_labeled_file, dtype={"chrom": str})
        
        # Load BAM files for read-level analysis
        self.cdna_bam = pysam.AlignmentFile(cdna_config.bam_file, "rb")
        self.drna_bam = pysam.AlignmentFile(drna_config.bam_file, "rb")
        
        # Results storage
        self.comparative_results = {}
        
        print(f"Loaded datasets:")
        print(f"  cDNA TSS: {len(self.cdna_tss)} sites")
        print(f"  cDNA TES: {len(self.cdna_tes)} sites")
        print(f"  dRNA TSS: {len(self.drna_tss)} sites")
        print(f"  dRNA TES: {len(self.drna_tes)} sites")
    
    def analyze_feature_distributions(self, site_type='tss', top_n_features=20):
        """Compare feature distributions between cDNA and dRNA."""
        print(f"\nAnalyzing feature distribution differences for {site_type.upper()}...")
        
        # Get datasets
        cdna_df = self.cdna_tss if site_type == 'tss' else self.cdna_tes
        drna_df = self.drna_tss if site_type == 'tss' else self.drna_tes
        
        # Load selected features for both datasets
        cdna_features = load_selected_features(self.cdna_config, site_type)
        drna_features = load_selected_features(self.drna_config, site_type)
        
        if not cdna_features or not drna_features:
            print(f"Warning: Missing selected features for {site_type}")
            return {}
        
        # Take top N features from each (they might be different)
        cdna_top_features = cdna_features[:top_n_features]
        drna_top_features = drna_features[:top_n_features]
        
        # Find common features for comparison
        common_features = list(set(cdna_top_features) & set(drna_top_features))
        
        # Also include features that exist in both datasets even if not in top N
        all_common_features = []
        for feat in set(cdna_features + drna_features):
            if feat in cdna_df.columns and feat in drna_df.columns:
                all_common_features.append(feat)
        
        print(f"  Found {len(common_features)} common top features")
        print(f"  Found {len(all_common_features)} total common features")
        
        # Analyze distributions for positive and negative sites separately
        results = {
            'common_top_features': common_features,
            'all_common_features': all_common_features,
            'positive_site_analysis': {},
            'negative_site_analysis': {},
            'feature_importance_comparison': {}
        }
        
        # Compare distributions for positive sites
        cdna_pos = cdna_df[cdna_df['label'] == 1]
        drna_pos = drna_df[drna_df['label'] == 1]
        
        print(f"  Positive sites - cDNA: {len(cdna_pos)}, dRNA: {len(drna_pos)}")
        
        results['positive_site_analysis'] = self._compare_feature_distributions(
            cdna_pos, drna_pos, all_common_features, 'positive_sites'
        )
        
        # Compare distributions for negative sites
        cdna_neg = cdna_df[cdna_df['label'] == 0]
        drna_neg = drna_df[drna_df['label'] == 0]
        
        print(f"  Negative sites - cDNA: {len(cdna_neg)}, dRNA: {len(drna_neg)}")
        
        results['negative_site_analysis'] = self._compare_feature_distributions(
            cdna_neg, drna_neg, all_common_features, 'negative_sites'
        )
        
        # Compare feature importance rankings
        results['feature_importance_comparison'] = self._compare_feature_importance(
            cdna_features, drna_features, site_type
        )
        
        self.comparative_results[f'{site_type}_feature_distributions'] = results
        return results
    
    def _compare_feature_distributions(self, cdna_df, drna_df, features, label):
        """Compare feature distributions between cDNA and dRNA datasets."""
        comparison_results = {}
        
        for feature in features:
            if feature not in cdna_df.columns or feature not in drna_df.columns:
                continue
                
            cdna_values = cdna_df[feature].dropna()
            drna_values = drna_df[feature].dropna()
            
            if len(cdna_values) == 0 or len(drna_values) == 0:
                continue
            
            # Statistical tests
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_pval = stats.ks_2samp(cdna_values, drna_values)
            
            # Mann-Whitney U test for median difference
            mw_stat, mw_pval = stats.mannwhitneyu(cdna_values, drna_values, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(cdna_values) - 1) * cdna_values.var() + 
                                (len(drna_values) - 1) * drna_values.var()) / 
                               (len(cdna_values) + len(drna_values) - 2))
            cohens_d = (cdna_values.mean() - drna_values.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Descriptive statistics
            comparison_results[feature] = {
                'cdna_mean': cdna_values.mean(),
                'cdna_std': cdna_values.std(),
                'cdna_median': cdna_values.median(),
                'cdna_q25': cdna_values.quantile(0.25),
                'cdna_q75': cdna_values.quantile(0.75),
                'drna_mean': drna_values.mean(),
                'drna_std': drna_values.std(),
                'drna_median': drna_values.median(),
                'drna_q25': drna_values.quantile(0.25),
                'drna_q75': drna_values.quantile(0.75),
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'mannwhitney_statistic': mw_stat,
                'mannwhitney_pvalue': mw_pval,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
            }
        
        return comparison_results
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _compare_feature_importance(self, cdna_features, drna_features, site_type):
        """Compare feature importance rankings between cDNA and dRNA."""
        # Create ranking dictionaries
        cdna_ranking = {feat: i for i, feat in enumerate(cdna_features)}
        drna_ranking = {feat: i for i, feat in enumerate(drna_features)}
        
        # Find common features and their rank differences
        common_features = set(cdna_features) & set(drna_features)
        
        ranking_comparison = {}
        for feature in common_features:
            cdna_rank = cdna_ranking[feature]
            drna_rank = drna_ranking[feature]
            rank_diff = drna_rank - cdna_rank  # Positive means worse in dRNA
            
            ranking_comparison[feature] = {
                'cdna_rank': cdna_rank,
                'drna_rank': drna_rank,
                'rank_difference': rank_diff,
                'rank_change_direction': 'worse_in_drna' if rank_diff > 0 else 'better_in_drna'
            }
        
        return {
            'common_features_count': len(common_features),
            'cdna_unique_features': set(cdna_features) - set(drna_features),
            'drna_unique_features': set(drna_features) - set(cdna_features),
            'ranking_comparison': ranking_comparison
        }
    
    def analyze_signal_to_noise_ratio(self, site_type='tss'):
        """Analyze signal-to-noise ratio differences between cDNA and dRNA."""
        print(f"\nAnalyzing signal-to-noise ratio for {site_type.upper()}...")
        
        # Get datasets and models
        cdna_df = self.cdna_tss if site_type == 'tss' else self.cdna_tes
        drna_df = self.drna_tss if site_type == 'tss' else self.drna_tes
        
        # Load pre-computed predictions
        cdna_predictions = self._load_predictions(site_type, 'cdna')
        drna_predictions = self._load_predictions(site_type, 'drna')
        
        if cdna_predictions is None or drna_predictions is None:
            print(f"Warning: Could not load models for {site_type}")
            return {}
        
        # Classify predictions into TP, FP, TN, FN
        cdna_analysis = self._analyze_prediction_quality(cdna_df, cdna_predictions, 'cdna')
        drna_analysis = self._analyze_prediction_quality(drna_df, drna_predictions, 'drna')
        
        # Compare signal-to-noise ratios
        results = {
            'cdna_analysis': cdna_analysis,
            'drna_analysis': drna_analysis,
            'signal_noise_comparison': self._compare_signal_noise(cdna_analysis, drna_analysis),
            'coverage_depth_analysis': self._analyze_coverage_impact(cdna_df, drna_df, site_type)
        }
        
        self.comparative_results[f'{site_type}_signal_noise'] = results
        return results
    
    def _load_predictions(self, site_type, dataset_name):
        """Load pre-computed predictions from predictions folder."""
        try:
            # Determine predictions directory and file
            predictions_dir = self.cdna_predictions_dir if dataset_name == 'cdna' else self.drna_predictions_dir
            predictions_file = os.path.join(predictions_dir, f"{site_type}_xgboost_predictions.csv")
            
            if not os.path.exists(predictions_file):
                print(f"Predictions file not found: {predictions_file}")
                return None
            
            # Load predictions
            pred_df = pd.read_csv(predictions_file, dtype={"chrom": str})
            
            # Get corresponding labeled data to extract true labels
            config = self.cdna_config if dataset_name == 'cdna' else self.drna_config
            labeled_df = pd.read_csv(config.tss_labeled_file if site_type == 'tss' else config.tes_labeled_file, 
                                   dtype={"chrom": str})
            
            # Merge predictions with true labels based on position
            merged_df = pred_df.merge(
                labeled_df[['chrom', 'position', 'label']], 
                on=['chrom', 'position'], 
                how='inner'
            )
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            y_pred = (merged_df['probability'] > 0.5).astype(int)
            y_prob = merged_df['probability'].values
            y_true = merged_df['label'].values
            
            # Get feature names (all columns except metadata)
            metadata_cols = ['site_type', 'chrom', 'position', 'probability', 'label']
            features_used = [col for col in merged_df.columns if col not in metadata_cols]
            
            print(f"Loaded {len(merged_df)} predictions for {dataset_name} {site_type}")
            print(f"Features used: {len(features_used)}")
            
            return {
                'predictions': y_pred,
                'probabilities': y_prob,
                'true_labels': y_true,
                'features_used': features_used,
                'merged_data': merged_df
            }
            
        except Exception as e:
            print(f"Error loading predictions for {dataset_name} {site_type}: {e}")
            return None
    
    def _analyze_prediction_quality(self, df, predictions, dataset_name):
        """Analyze prediction quality and classify into TP, FP, TN, FN."""
        y_true = predictions['true_labels']
        y_pred = predictions['predictions']
        y_prob = predictions['probabilities']
        
        # Classify predictions
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        # Get feature values for each category from the merged predictions data
        features = predictions['features_used']
        merged_data = predictions['merged_data']
        
        tp_features = merged_data.loc[tp_mask, features] if tp_mask.sum() > 0 else pd.DataFrame()
        fp_features = merged_data.loc[fp_mask, features] if fp_mask.sum() > 0 else pd.DataFrame()
        tn_features = merged_data.loc[tn_mask, features] if tn_mask.sum() > 0 else pd.DataFrame()
        fn_features = merged_data.loc[fn_mask, features] if fn_mask.sum() > 0 else pd.DataFrame()
        
        # Calculate signal-to-noise metrics
        analysis = {
            'dataset': dataset_name,
            'n_tp': tp_mask.sum(),
            'n_fp': fp_mask.sum(),
            'n_tn': tn_mask.sum(),
            'n_fn': fn_mask.sum(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': sum(tp_mask) / max(1, sum(tp_mask) + sum(fp_mask)),
            'recall': sum(tp_mask) / max(1, sum(tp_mask) + sum(fn_mask)),
            'tp_prob_mean': y_prob[tp_mask].mean() if tp_mask.sum() > 0 else 0,
            'fp_prob_mean': y_prob[fp_mask].mean() if fp_mask.sum() > 0 else 0,
            'tn_prob_mean': y_prob[tn_mask].mean() if tn_mask.sum() > 0 else 0,
            'fn_prob_mean': y_prob[fn_mask].mean() if fn_mask.sum() > 0 else 0,
        }
        
        # Calculate feature-wise signal-to-noise ratios (TP vs FP)
        if len(tp_features) > 0 and len(fp_features) > 0:
            feature_snr = {}
            for feature in features:
                if feature in tp_features.columns and feature in fp_features.columns:
                    tp_vals = tp_features[feature].dropna()
                    fp_vals = fp_features[feature].dropna()
                    
                    if len(tp_vals) > 0 and len(fp_vals) > 0:
                        # Signal-to-noise as ratio of means
                        tp_mean = tp_vals.mean()
                        fp_mean = fp_vals.mean()
                        
                        # Avoid division by zero
                        if abs(fp_mean) > 1e-10:
                            snr = abs(tp_mean) / abs(fp_mean) if abs(fp_mean) > 0 else float('inf')
                        else:
                            snr = float('inf') if abs(tp_mean) > 1e-10 else 1.0
                        
                        # Also calculate separation (difference in means relative to pooled std)
                        pooled_std = np.sqrt((tp_vals.var() + fp_vals.var()) / 2)
                        separation = abs(tp_mean - fp_mean) / pooled_std if pooled_std > 0 else 0
                        
                        feature_snr[feature] = {
                            'tp_mean': tp_mean,
                            'fp_mean': fp_mean,
                            'tp_std': tp_vals.std(),
                            'fp_std': fp_vals.std(),
                            'snr_ratio': snr,
                            'separation': separation
                        }
            
            analysis['feature_snr'] = feature_snr
        
        return analysis
    
    def _compare_signal_noise(self, cdna_analysis, drna_analysis):
        """Compare signal-to-noise ratios between cDNA and dRNA."""
        comparison = {
            'overall_metrics': {
                'cdna_accuracy': cdna_analysis['accuracy'],
                'drna_accuracy': drna_analysis['accuracy'],
                'accuracy_difference': cdna_analysis['accuracy'] - drna_analysis['accuracy'],
                'cdna_precision': cdna_analysis['precision'],
                'drna_precision': drna_analysis['precision'],
                'precision_difference': cdna_analysis['precision'] - drna_analysis['precision'],
            }
        }
        
        # Compare confidence scores
        comparison['confidence_analysis'] = {
            'cdna_tp_confidence': cdna_analysis['tp_prob_mean'],
            'drna_tp_confidence': drna_analysis['tp_prob_mean'],
            'cdna_fp_confidence': cdna_analysis['fp_prob_mean'],
            'drna_fp_confidence': drna_analysis['fp_prob_mean'],
            'cdna_confidence_gap': cdna_analysis['tp_prob_mean'] - cdna_analysis['fp_prob_mean'],
            'drna_confidence_gap': drna_analysis['tp_prob_mean'] - drna_analysis['fp_prob_mean'],
        }
        
        # Compare feature-level signal-to-noise ratios
        if 'feature_snr' in cdna_analysis and 'feature_snr' in drna_analysis:
            feature_snr_comparison = {}
            
            common_features = set(cdna_analysis['feature_snr'].keys()) & set(drna_analysis['feature_snr'].keys())
            
            for feature in common_features:
                cdna_snr = cdna_analysis['feature_snr'][feature]
                drna_snr = drna_analysis['feature_snr'][feature]
                
                feature_snr_comparison[feature] = {
                    'cdna_snr': cdna_snr['snr_ratio'],
                    'drna_snr': drna_snr['snr_ratio'],
                    'snr_ratio_difference': cdna_snr['snr_ratio'] - drna_snr['snr_ratio'],
                    'cdna_separation': cdna_snr['separation'],
                    'drna_separation': drna_snr['separation'],
                    'separation_difference': cdna_snr['separation'] - drna_snr['separation'],
                }
            
            comparison['feature_snr_comparison'] = feature_snr_comparison
        
        return comparison
    
    def _analyze_coverage_impact(self, cdna_df, drna_df, site_type):
        """Analyze how coverage depth affects performance differently in cDNA vs dRNA."""
        coverage_features = ['coverage_before', 'coverage_after', 'total_reads']
        
        # Calculate combined coverage
        cdna_coverage = cdna_df[coverage_features].sum(axis=1)
        drna_coverage = drna_df[coverage_features].sum(axis=1)
        
        # Define coverage bins
        cdna_coverage_bins = pd.qcut(cdna_coverage, q=4, labels=['low', 'medium_low', 'medium_high', 'high'])
        drna_coverage_bins = pd.qcut(drna_coverage, q=4, labels=['low', 'medium_low', 'medium_high', 'high'])
        
        coverage_analysis = {
            'cdna_coverage_stats': {
                'mean': cdna_coverage.mean(),
                'median': cdna_coverage.median(),
                'std': cdna_coverage.std(),
                'min': cdna_coverage.min(),
                'max': cdna_coverage.max()
            },
            'drna_coverage_stats': {
                'mean': drna_coverage.mean(),
                'median': drna_coverage.median(),
                'std': drna_coverage.std(),
                'min': drna_coverage.min(),
                'max': drna_coverage.max()
            }
        }
        
        # Test for coverage distribution differences
        ks_stat, ks_pval = stats.ks_2samp(cdna_coverage, drna_coverage)
        coverage_analysis['coverage_distribution_test'] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval
        }
        
        return coverage_analysis
    
    def analyze_five_prime_degradation(self, site_type='tss'):
        """Analyze 5' degradation impact using coverage gradients and soft-clipping patterns."""
        print(f"\nAnalyzing 5' degradation impact for {site_type.upper()}...")
        
        cdna_df = self.cdna_tss if site_type == 'tss' else self.cdna_tes
        drna_df = self.drna_tss if site_type == 'tss' else self.drna_tes
        
        results = {
            'coverage_gradient_analysis': self._analyze_coverage_gradients(cdna_df, drna_df, site_type),
            'softclip_pattern_analysis': self._analyze_softclip_patterns(cdna_df, drna_df, site_type),
            'read_start_precision_analysis': self._analyze_read_start_precision(cdna_df, drna_df, site_type)
        }
        
        self.comparative_results[f'{site_type}_five_prime_degradation'] = results
        return results
    
    def _analyze_coverage_gradients(self, cdna_df, drna_df, site_type):
        """Analyze coverage gradients as proxy for 5' completeness."""
        # Calculate coverage ratios and deltas
        cdna_coverage_ratio = cdna_df['coverage_after'] / (cdna_df['coverage_before'] + 1e-10)
        drna_coverage_ratio = drna_df['coverage_after'] / (drna_df['coverage_before'] + 1e-10)
        
        cdna_coverage_delta = cdna_df['coverage_after'] - cdna_df['coverage_before']
        drna_coverage_delta = drna_df['coverage_after'] - drna_df['coverage_before']
        
        # For TSS, we expect higher coverage after (downstream) for complete 5' ends
        # For TES, we expect higher coverage before (upstream) for complete 3' ends
        
        analysis = {
            'coverage_ratio_comparison': {
                'cdna_mean_ratio': cdna_coverage_ratio.mean(),
                'drna_mean_ratio': drna_coverage_ratio.mean(),
                'ratio_difference': cdna_coverage_ratio.mean() - drna_coverage_ratio.mean(),
                'ks_test': stats.ks_2samp(cdna_coverage_ratio, drna_coverage_ratio)
            },
            'coverage_delta_comparison': {
                'cdna_mean_delta': cdna_coverage_delta.mean(),
                'drna_mean_delta': drna_coverage_delta.mean(),
                'delta_difference': cdna_coverage_delta.mean() - drna_coverage_delta.mean(),
                'ks_test': stats.ks_2samp(cdna_coverage_delta, drna_coverage_delta)
            }
        }
        
        return analysis
    
    def _analyze_softclip_patterns(self, cdna_df, drna_df, site_type):
        """Analyze soft-clipping patterns as indicator of degradation."""
        # Get soft-clipping related features
        softclip_features = [col for col in cdna_df.columns if 'soft_clip' in col.lower()]
        
        if not softclip_features:
            return {'error': 'No soft-clipping features found'}
        
        analysis = {}
        
        for feature in softclip_features:
            if feature in drna_df.columns:
                cdna_vals = cdna_df[feature].dropna()
                drna_vals = drna_df[feature].dropna()
                
                if len(cdna_vals) > 0 and len(drna_vals) > 0:
                    # Statistical comparison
                    ks_stat, ks_pval = stats.ks_2samp(cdna_vals, drna_vals)
                    mw_stat, mw_pval = stats.mannwhitneyu(cdna_vals, drna_vals, alternative='two-sided')
                    
                    analysis[feature] = {
                        'cdna_mean': cdna_vals.mean(),
                        'cdna_median': cdna_vals.median(),
                        'cdna_std': cdna_vals.std(),
                        'drna_mean': drna_vals.mean(),
                        'drna_median': drna_vals.median(),
                        'drna_std': drna_vals.std(),
                        'mean_difference': drna_vals.mean() - cdna_vals.mean(),
                        'median_difference': drna_vals.median() - cdna_vals.median(),
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pval,
                        'mannwhitney_pvalue': mw_pval
                    }
        
        return analysis
    
    def _analyze_read_start_precision(self, cdna_df, drna_df, site_type):
        """Analyze read start/end precision using entropy features."""
        entropy_features = [col for col in cdna_df.columns if 'entropy' in col.lower()]
        
        if not entropy_features:
            return {'error': 'No entropy features found'}
        
        analysis = {}
        
        for feature in entropy_features:
            if feature in drna_df.columns:
                cdna_vals = cdna_df[feature].dropna()
                drna_vals = drna_df[feature].dropna()
                
                if len(cdna_vals) > 0 and len(drna_vals) > 0:
                    # Higher entropy = less precise read starts/ends
                    analysis[feature] = {
                        'cdna_mean_entropy': cdna_vals.mean(),
                        'drna_mean_entropy': drna_vals.mean(),
                        'entropy_difference': drna_vals.mean() - cdna_vals.mean(),
                        'precision_interpretation': 'dRNA less precise' if drna_vals.mean() > cdna_vals.mean() else 'dRNA more precise',
                        'ks_test': stats.ks_2samp(cdna_vals, drna_vals)
                    }
        
        return analysis
    
    def apply_statistical_corrections(self):
        """Apply Bonferroni correction to all p-values in the results."""
        print("\nApplying Bonferroni correction to p-values...")
        
        # Collect all p-values
        all_pvalues = []
        pvalue_locations = []
        
        def collect_pvalues(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.endswith('_pvalue') or key.endswith('pvalue'):
                        if isinstance(value, (int, float)) and 0 <= value <= 1:
                            all_pvalues.append(value)
                            pvalue_locations.append(f"{path}.{key}")
                    elif isinstance(value, tuple) and len(value) == 2:
                        # Handle scipy test results (statistic, pvalue)
                        if key.endswith('_test'):
                            all_pvalues.append(value[1])
                            pvalue_locations.append(f"{path}.{key}[1]")
                    else:
                        collect_pvalues(value, f"{path}.{key}" if path else key)
        
        collect_pvalues(self.comparative_results)
        
        if not all_pvalues:
            print("No p-values found for correction")
            return
        
        # Apply Bonferroni correction
        n_tests = len(all_pvalues)
        corrected_alpha = 0.05 / n_tests
        
        print(f"Found {n_tests} statistical tests")
        print(f"Bonferroni corrected alpha: {corrected_alpha:.6f}")
        
        # Store correction info
        self.comparative_results['statistical_correction'] = {
            'method': 'bonferroni',
            'original_alpha': 0.05,
            'corrected_alpha': corrected_alpha,
            'n_tests': n_tests,
            'significant_tests': sum(1 for p in all_pvalues if p < corrected_alpha)
        }
    
    def generate_comparative_report(self, output_dir):
        """Generate comprehensive comparative analysis report."""
        print(f"\nGenerating comparative analysis report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply statistical corrections first
        self.apply_statistical_corrections()
        
        # Generate individual analysis reports
        for analysis_type, results in self.comparative_results.items():
            if analysis_type == 'statistical_correction':
                continue
                
            # Save detailed results as CSV
            self._save_analysis_results(results, analysis_type, output_dir)
        
        # Create visualizations
        self._create_comparative_visualizations(output_dir)
        
        # Generate comprehensive text report
        self._create_comprehensive_report(output_dir)
        
        print(f"Comparative analysis complete! Results saved to {output_dir}")
    
    def _save_analysis_results(self, results, analysis_type, output_dir):
        """Save analysis results as structured CSV files."""
        try:
            # Convert nested results to flat structure for CSV
            if 'positive_site_analysis' in results:
                # Feature distribution analysis
                pos_df = pd.DataFrame(results['positive_site_analysis']).T
                neg_df = pd.DataFrame(results['negative_site_analysis']).T
                
                pos_df.to_csv(os.path.join(output_dir, f"{analysis_type}_positive_sites.csv"))
                neg_df.to_csv(os.path.join(output_dir, f"{analysis_type}_negative_sites.csv"))
            
            elif 'coverage_gradient_analysis' in results:
                # 5' degradation analysis
                for sub_analysis, sub_results in results.items():
                    if isinstance(sub_results, dict):
                        df = pd.DataFrame([sub_results]).T
                        df.to_csv(os.path.join(output_dir, f"{analysis_type}_{sub_analysis}.csv"))
            
            else:
                # General case - try to convert to DataFrame
                try:
                    df = pd.DataFrame([results]).T
                    df.to_csv(os.path.join(output_dir, f"{analysis_type}_results.csv"))
                except:
                    # Save as JSON if DataFrame conversion fails
                    import json
                    with open(os.path.join(output_dir, f"{analysis_type}_results.json"), 'w') as f:
                        json.dump(results, f, indent=2, default=str)
        
        except Exception as e:
            print(f"Warning: Could not save {analysis_type} results: {e}")
    
    def _create_comparative_visualizations(self, output_dir):
        """Create comprehensive visualization plots."""
        print("Creating comparative visualizations...")
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for site_type in ['tss', 'tes']:
            self._create_feature_distribution_plots(site_type, output_dir)
            self._create_signal_noise_plots(site_type, output_dir)
    
    def _create_feature_distribution_plots(self, site_type, output_dir):
        """Create feature distribution comparison plots."""
        analysis_key = f'{site_type}_feature_distributions'
        if analysis_key not in self.comparative_results:
            return
        
        results = self.comparative_results[analysis_key]
        
        # Plot top features with largest differences
        pos_analysis = results.get('positive_site_analysis', {})
        neg_analysis = results.get('negative_site_analysis', {})
        
        if not pos_analysis and not neg_analysis:
            return
        
        # Find features with largest effect sizes
        all_features = set(pos_analysis.keys()) | set(neg_analysis.keys())
        feature_effects = []
        
        for feature in all_features:
            pos_effect = abs(pos_analysis.get(feature, {}).get('cohens_d', 0))
            neg_effect = abs(neg_analysis.get(feature, {}).get('cohens_d', 0))
            max_effect = max(pos_effect, neg_effect)
            feature_effects.append((feature, max_effect))
        
        # Sort by effect size and take top 10
        top_features = sorted(feature_effects, key=lambda x: x[1], reverse=True)[:10]
        
        if not top_features:
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.suptitle(f'{site_type.upper()} Feature Distribution Comparisons (Top 10 by Effect Size)', fontsize=16)
        
        for i, (feature, effect_size) in enumerate(top_features):
            row, col = i // 5, i % 5
            ax = axes[row, col]
            
            # Get data for plotting
            cdna_df = self.cdna_tss if site_type == 'tss' else self.cdna_tes
            drna_df = self.drna_tss if site_type == 'tss' else self.drna_tes
            
            if feature in cdna_df.columns and feature in drna_df.columns:
                cdna_vals = cdna_df[feature].dropna()
                drna_vals = drna_df[feature].dropna()
                
                # Create histograms
                ax.hist(cdna_vals, alpha=0.6, label='cDNA', bins=30, density=True)
                ax.hist(drna_vals, alpha=0.6, label='dRNA', bins=30, density=True)
                ax.set_title(f'{feature}\n(Effect size: {effect_size:.3f})')
                ax.legend()
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{site_type}_feature_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_signal_noise_plots(self, site_type, output_dir):
        """Create signal-to-noise comparison plots."""
        analysis_key = f'{site_type}_signal_noise'
        if analysis_key not in self.comparative_results:
            return
        
        results = self.comparative_results[analysis_key]
        
        # Create performance comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{site_type.upper()} Performance Comparison: cDNA vs dRNA', fontsize=16)
        
        # Plot 1: Overall metrics
        metrics = ['accuracy', 'precision']
        cdna_vals = [results['signal_noise_comparison']['overall_metrics'][f'cdna_{m}'] for m in metrics]
        drna_vals = [results['signal_noise_comparison']['overall_metrics'][f'drna_{m}'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0].bar(x - width/2, cdna_vals, width, label='cDNA', alpha=0.8)
        axes[0].bar(x + width/2, drna_vals, width, label='dRNA', alpha=0.8)
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Overall Performance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.capitalize() for m in metrics])
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Confidence analysis
        conf_analysis = results['signal_noise_comparison']['confidence_analysis']
        conf_metrics = ['tp_confidence', 'fp_confidence']
        cdna_conf = [conf_analysis[f'cdna_{m}'] for m in conf_metrics]
        drna_conf = [conf_analysis[f'drna_{m}'] for m in conf_metrics]
        
        x = np.arange(len(conf_metrics))
        axes[1].bar(x - width/2, cdna_conf, width, label='cDNA', alpha=0.8)
        axes[1].bar(x + width/2, drna_conf, width, label='dRNA', alpha=0.8)
        axes[1].set_xlabel('Prediction Type')
        axes[1].set_ylabel('Mean Confidence')
        axes[1].set_title('Prediction Confidence')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['True Positive', 'False Positive'])
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        # Plot 3: Confidence gap
        cdna_gap = conf_analysis['cdna_confidence_gap']
        drna_gap = conf_analysis['drna_confidence_gap']
        
        axes[2].bar(['cDNA', 'dRNA'], [cdna_gap, drna_gap], alpha=0.8)
        axes[2].set_ylabel('Confidence Gap (TP - FP)')
        axes[2].set_title('Model Confidence Gap')
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{site_type}_signal_noise_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_report(self, output_dir):
        """Create comprehensive text report summarizing all findings."""
        report_file = os.path.join(output_dir, "comprehensive_comparative_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE COMPARATIVE ANALYSIS REPORT\n")
            f.write("cDNA vs dRNA Performance Analysis\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write("This report analyzes why dRNA performs worse than cDNA by examining:\n")
            f.write("1. Feature distribution differences\n")
            f.write("2. Signal-to-noise ratio variations\n")
            f.write("3. 5' degradation impact\n\n")
            
            # Statistical correction summary
            if 'statistical_correction' in self.comparative_results:
                stats_info = self.comparative_results['statistical_correction']
                f.write(f"Statistical Testing: {stats_info['n_tests']} tests performed\n")
                f.write(f"Bonferroni corrected α: {stats_info['corrected_alpha']:.6f}\n")
                f.write(f"Significant tests: {stats_info['significant_tests']}\n\n")
            
            # Detailed analysis for each site type
            for site_type in ['tss', 'tes']:
                f.write(f"\n{site_type.upper()} ANALYSIS\n")
                f.write("=" * 30 + "\n")
                
                # Feature distribution analysis
                self._write_feature_analysis(f, site_type)
                
                # Signal-to-noise analysis
                self._write_signal_noise_analysis(f, site_type)
                
                # 5' degradation analysis
                self._write_degradation_analysis(f, site_type)
            
            # Overall conclusions
            f.write("\nOVERALL CONCLUSIONS\n")
            f.write("=" * 30 + "\n")
            self._write_conclusions(f)
        
        print(f"Comprehensive report saved to {report_file}")
    
    def _write_feature_analysis(self, f, site_type):
        """Write feature distribution analysis to report."""
        analysis_key = f'{site_type}_feature_distributions'
        if analysis_key not in self.comparative_results:
            f.write(f"Feature distribution analysis not available for {site_type}\n")
            return
        
        results = self.comparative_results[analysis_key]
        f.write(f"\nFeature Distribution Analysis ({site_type.upper()})\n")
        f.write("-" * 40 + "\n")
        
        # Positive sites analysis
        pos_analysis = results.get('positive_site_analysis', {})
        if pos_analysis:
            f.write("Positive Sites - Features with largest differences:\n")
            
            # Sort by effect size
            sorted_features = sorted(pos_analysis.items(), 
                                   key=lambda x: abs(x[1].get('cohens_d', 0)), 
                                   reverse=True)[:5]
            
            for feature, stats in sorted_features:
                effect_size = stats.get('cohens_d', 0)
                interpretation = stats.get('effect_size_interpretation', 'unknown')
                f.write(f"  {feature}: Cohen's d = {effect_size:.3f} ({interpretation})\n")
                f.write(f"    cDNA mean: {stats.get('cdna_mean', 0):.3f}, dRNA mean: {stats.get('drna_mean', 0):.3f}\n")
        
        # Feature importance comparison
        importance_comp = results.get('feature_importance_comparison', {})
        if importance_comp:
            f.write(f"\nFeature Importance Ranking:\n")
            f.write(f"  Common features: {importance_comp.get('common_features_count', 0)}\n")
            f.write(f"  cDNA unique: {len(importance_comp.get('cdna_unique_features', []))}\n")
            f.write(f"  dRNA unique: {len(importance_comp.get('drna_unique_features', []))}\n")
    
    def _write_signal_noise_analysis(self, f, site_type):
        """Write signal-to-noise analysis to report."""
        analysis_key = f'{site_type}_signal_noise'
        if analysis_key not in self.comparative_results:
            f.write(f"Signal-to-noise analysis not available for {site_type}\n")
            return
        
        results = self.comparative_results[analysis_key]
        f.write(f"\nSignal-to-Noise Analysis ({site_type.upper()})\n")
        f.write("-" * 40 + "\n")
        
        # Overall performance
        overall = results.get('signal_noise_comparison', {}).get('overall_metrics', {})
        if overall:
            f.write("Performance Comparison:\n")
            f.write(f"  Accuracy: cDNA {overall.get('cdna_accuracy', 0):.3f} vs dRNA {overall.get('drna_accuracy', 0):.3f}\n")
            f.write(f"  Precision: cDNA {overall.get('cdna_precision', 0):.3f} vs dRNA {overall.get('drna_precision', 0):.3f}\n")
            f.write(f"  Accuracy difference: {overall.get('accuracy_difference', 0):.3f}\n")
        
        # Confidence analysis
        conf = results.get('signal_noise_comparison', {}).get('confidence_analysis', {})
        if conf:
            f.write("\nModel Confidence:\n")
            f.write(f"  cDNA confidence gap (TP-FP): {conf.get('cdna_confidence_gap', 0):.3f}\n")
            f.write(f"  dRNA confidence gap (TP-FP): {conf.get('drna_confidence_gap', 0):.3f}\n")
    
    def _write_degradation_analysis(self, f, site_type):
        """Write 5' degradation analysis to report."""
        analysis_key = f'{site_type}_five_prime_degradation'
        if analysis_key not in self.comparative_results:
            f.write(f"5' degradation analysis not available for {site_type}\n")
            return
        
        results = self.comparative_results[analysis_key]
        f.write(f"\n5' Degradation Impact Analysis ({site_type.upper()})\n")
        f.write("-" * 40 + "\n")
        
        # Coverage gradient analysis
        cov_grad = results.get('coverage_gradient_analysis', {})
        if cov_grad:
            ratio_comp = cov_grad.get('coverage_ratio_comparison', {})
            delta_comp = cov_grad.get('coverage_delta_comparison', {})
            
            f.write("Coverage Gradient Analysis:\n")
            if ratio_comp:
                f.write(f"  Coverage ratio - cDNA: {ratio_comp.get('cdna_mean_ratio', 0):.3f}, dRNA: {ratio_comp.get('drna_mean_ratio', 0):.3f}\n")
                f.write(f"  Ratio difference: {ratio_comp.get('ratio_difference', 0):.3f}\n")
        
        # Soft-clipping analysis
        softclip = results.get('softclip_pattern_analysis', {})
        if softclip and not softclip.get('error'):
            f.write("\nSoft-clipping Pattern Differences:\n")
            for feature, stats in list(softclip.items())[:3]:  # Top 3 features
                f.write(f"  {feature}: dRNA mean {stats.get('drna_mean', 0):.3f} vs cDNA mean {stats.get('cdna_mean', 0):.3f}\n")
    
    def _write_conclusions(self, f):
        """Write overall conclusions to the report."""
        f.write("Based on the comparative analysis:\n\n")
        
        f.write("1. FEATURE DISTRIBUTION DIFFERENCES:\n")
        f.write("   - Identify features with large effect sizes between cDNA and dRNA\n")
        f.write("   - Focus on soft-clipping and coverage-related features\n\n")
        
        f.write("2. SIGNAL-TO-NOISE RATIO:\n")
        f.write("   - Compare model confidence gaps between true and false positives\n")
        f.write("   - Assess prediction quality differences\n\n")
        
        f.write("3. 5' DEGRADATION IMPACT:\n")
        f.write("   - Analyze coverage gradients and soft-clipping patterns\n")
        f.write("   - Evaluate read start precision differences\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("- Focus on features showing largest differences\n")
        f.write("- Consider dRNA-specific preprocessing or normalization\n")
        f.write("- Develop technology-specific models if differences are substantial\n")
    
    def run_full_comparative_analysis(self, output_dir):
        """Run complete comparative analysis."""
        print("Starting comprehensive comparative analysis: cDNA vs dRNA")
        print("=" * 60)
        
        # Run analyses for both site types
        for site_type in ['tss', 'tes']:
            print(f"\n=== Analyzing {site_type.upper()} ===")
            
            try:
                # Feature distribution analysis
                self.analyze_feature_distributions(site_type)
                
                # Signal-to-noise analysis
                self.analyze_signal_to_noise_ratio(site_type)
                
                # 5' degradation analysis
                self.analyze_five_prime_degradation(site_type)
                
            except Exception as e:
                print(f"Error in {site_type} analysis: {e}")
                continue
        
        # Generate comprehensive report
        self.generate_comparative_report(output_dir)
        
        # Close BAM files
        self.cdna_bam.close()
        self.drna_bam.close()
        
        print(f"\nComparative analysis complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform comprehensive comparative analysis between cDNA and dRNA"
    )
    parser.add_argument("--cdna-config", required=True,
                       help="Path to cDNA project configuration file")
    parser.add_argument("--drna-config", required=True,
                       help="Path to dRNA project configuration file")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for comparative analysis results")
    
    args = parser.parse_args()
    
    # Load configurations
    cdna_config = load_config(args.cdna_config)
    drna_config = load_config(args.drna_config)
    
    print(f"cDNA predictions directory: {cdna_config.predictions_output_dir}")
    print(f"dRNA predictions directory: {drna_config.predictions_output_dir}")
    
    # Create analyzer
    analyzer = ComparativeAnalyzer(
        cdna_config=cdna_config,
        drna_config=drna_config
    )
    
    # Run full analysis
    analyzer.run_full_comparative_analysis(args.output_dir)


if __name__ == "__main__":
    main()
