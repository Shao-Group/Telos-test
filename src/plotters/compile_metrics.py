#!/usr/bin/env python3
"""
Script to compile all metrics from training and test datasets into separate CSV files
that can be used for LaTeX table generation.

This script uses the config files from project_config/ to locate the metrics directories
and creates separate tables for training and test datasets.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import ast
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

def parse_confusion_matrix(confusion_str: str) -> Dict[str, int]:
    """Parse confusion matrix string and extract TP, TN, FP, FN values."""
    try:
        # Parse the string representation of the matrix
        matrix = ast.literal_eval(confusion_str)
        tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        return {
            'TP': tp,
            'TN': tn, 
            'FP': fp,
            'FN': fn
        }
    except:
        return {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

def load_configs_from_directory(config_dir: str) -> Tuple[List[Config], List[Config]]:
    """
    Load all config files and separate them into training and test configs.
    
    Args:
        config_dir: Directory containing config pickle files
        
    Returns:
        Tuple of (train_configs, test_configs)
    """
    train_configs = []
    test_configs = []
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('_config.pkl')]
    
    for config_file in config_files:
        try:
            config_path = os.path.join(config_dir, config_file)
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            # Check if config has metrics_output_dir attribute (some older configs might not)
            if not hasattr(config, 'metrics_output_dir'):
                print(f"Warning: Config {config_file} doesn't have metrics_output_dir, skipping...")
                continue
                
            # Determine if it's train or test based on output_dir
            if 'train_output' in config.output_dir:
                train_configs.append(config)
            elif 'test_output' in config.output_dir:
                test_configs.append(config)
            else:
                print(f"Warning: Could not determine data type for {config_file} (output_dir: {config.output_dir})")
                
        except Exception as e:
            print(f"Error loading config {config_file}: {e}")
            continue
    
    return train_configs, test_configs

def find_metrics_files_from_configs(configs: List[Config]) -> List[Tuple[str, Dict]]:
    """
    Find all metrics CSV files based on config objects.
    
    Args:
        configs: List of Config objects
        
    Returns:
        List of tuples (file_path, dataset_info)
    """
    metrics_files = []
    
    # Define the expected metric file patterns
    tasks = ['tss', 'tes']
    models = ['randomforest', 'xgboost']
    
    for config in configs:
        if not os.path.exists(config.metrics_output_dir):
            print(f"Warning: Metrics directory {config.metrics_output_dir} does not exist, skipping...")
            continue
            
        # Parse dataset info from data_name
        data_name = config.data_name
        
        # Split dataset and assembler from data_name
        if '_' in data_name:
            # Find the last underscore to split dataset from assembler
            last_underscore = data_name.rfind('_')
            dataset = data_name[:last_underscore]
            assembler = data_name[last_underscore + 1:]
        else:
            dataset = data_name
            assembler = 'unknown'
        
        # Determine data type from output directory
        data_type = 'train' if 'train_output' in config.output_dir else 'test'
        
        # Look for all expected metric files
        for task in tasks:
            for model in models:
                metric_filename = f"{task}_{model}_metrics.csv"
                metric_filepath = os.path.join(config.metrics_output_dir, metric_filename)
                
                if os.path.exists(metric_filepath):
                    dataset_info = {
                        'data_type': data_type,
                        'dataset': dataset,
                        'assembler': assembler,
                        'task': task,
                        'model': model,
                        'data_name': data_name
                    }
                    metrics_files.append((metric_filepath, dataset_info))
                else:
                    print(f"Warning: Expected metrics file not found: {metric_filepath}")
    
    return metrics_files

def load_metrics_file(file_path: str) -> Dict[str, float]:
    """Load a single metrics CSV file and return metrics as dictionary."""
    try:
        df = pd.read_csv(file_path)
        metrics = {}
        
        for _, row in df.iterrows():
            metric_name = row['metric']
            value = row['value']
            
            if metric_name == 'confusion_matrix':
                # Parse confusion matrix and add individual components
                cm_values = parse_confusion_matrix(value)
                metrics.update(cm_values)
            else:
                # Convert to float if possible
                try:
                    metrics[metric_name] = float(value)
                except (ValueError, TypeError):
                    metrics[metric_name] = value
        
        return metrics
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def compile_metrics_from_configs(configs: List[Config], data_type: str) -> pd.DataFrame:
    """
    Compile all metrics from the specified configs into a single DataFrame.
    
    Args:
        configs: List of Config objects to process
        data_type: Type of data ('train' or 'test')
        
    Returns:
        DataFrame with compiled metrics
    """
    print(f"Finding {data_type} metrics files...")
    metrics_files = find_metrics_files_from_configs(configs)
    
    if not metrics_files:
        print(f"No {data_type} metrics files found!")
        return pd.DataFrame()
    
    print(f"Found {len(metrics_files)} {data_type} metrics files")
    
    all_data = []
    
    for file_path, dataset_info in metrics_files:
        print(f"Processing: {file_path}")
        metrics = load_metrics_file(file_path)
        
        if metrics:
            # Combine dataset info with metrics
            row_data = {**dataset_info, **metrics}
            all_data.append(row_data)
    
    if not all_data:
        print(f"No valid {data_type} metrics data found!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns to put identifying information first
    id_cols = ['data_type', 'dataset', 'assembler', 'task', 'model', 'data_name']
    metric_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + sorted(metric_cols)]
    
    return df

def format_for_latex(df: pd.DataFrame, output_file: str) -> None:
    """
    Format the DataFrame for LaTeX table generation and save to file.
    For LaTeX output, only include XGBoost results with specific metrics.
    Uses multirow formatting for datasets with proper grouping.
    
    Args:
        df: DataFrame with compiled metrics
        output_file: Path to save the LaTeX-formatted file
    """
    # Round numerical columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_latex = df.copy()
    
    for col in numeric_cols:
        if col not in ['TP', 'TN', 'FP', 'FN']:  # Keep confusion matrix values as integers
            df_latex[col] = df_latex[col].round(3)
    
    # Save complete CSV file
    df_latex.to_csv(output_file, index=False)
    
    # Filter for LaTeX table: only XGBoost results
    df_xgboost = df_latex[df_latex['model'] == 'xgboost'].copy()
    
    if df_xgboost.empty:
        print("Warning: No XGBoost results found for LaTeX table")
        return
    
    # Select only the specified metrics for LaTeX output
    latex_metrics = ['auc', 'aupr', 'precision', 'recall', 'f1']
    id_cols = ['dataset', 'assembler', 'task']
    
    # Check which metrics are available
    available_metrics = [col for col in latex_metrics if col in df_xgboost.columns]
    if not available_metrics:
        print("Warning: None of the specified metrics (auc, aupr, precision, recall, f1) found in data")
        return
    
    # Create LaTeX-specific DataFrame with selected columns
    latex_cols = id_cols + available_metrics
    df_latex_filtered = df_xgboost[latex_cols].copy()
    
    # Sort data for proper grouping: by dataset, then assembler, then task (TSS first, then TES)
    df_latex_filtered['task_order'] = df_latex_filtered['task'].map({'tss': 0, 'tes': 1})
    df_latex_filtered = df_latex_filtered.sort_values(['dataset', 'assembler', 'task_order'])
    df_latex_filtered = df_latex_filtered.drop('task_order', axis=1)
    
    # Create a LaTeX table snippet
    latex_file = output_file.replace('.csv', '_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("% LaTeX table snippet - XGBoost Results Only\n")
        f.write("% Metrics: " + ", ".join(available_metrics) + "\n")
        f.write("% Use \\input{" + os.path.basename(latex_file) + "} in your LaTeX document\n\n")
        
        # Create LaTeX table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{XGBoost Model Performance Metrics}\n")
        f.write("\\label{tab:xgboost_metrics}\n")
        
        # Column alignment: left for dataset, left for assembler, left for task, center for metrics
        col_alignment = "lll" + "c" * len(available_metrics)
        f.write("\\begin{tabular}{" + col_alignment + "}\n")
        f.write("\\toprule\n")
        
        # Header
        header_parts = ["Dataset", "Assembler", "Task"]
        for col in available_metrics:
            if col == 'auc':
                header_parts.append('AUC')
            elif col == 'aupr':
                header_parts.append('AUPR')
            elif col == 'precision':
                header_parts.append('Precision')
            elif col == 'recall':
                header_parts.append('Recall')
            elif col == 'f1':
                header_parts.append('F1-Score')
            else:
                header_parts.append(col.replace('_', '\\_').title())
        
        f.write(" & ".join(header_parts) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Generate data rows with multirow formatting
        current_dataset = None
        dataset_row_count = 0
        
        # First pass: count rows per dataset
        dataset_counts = df_latex_filtered.groupby('dataset').size().to_dict()
        
        for idx, (_, row) in enumerate(df_latex_filtered.iterrows()):
            dataset = row['dataset']
            assembler = row['assembler']
            task = row['task'].upper()
            
            # Format dataset name for display
            dataset_display = dataset.replace('_', '\\_')
            if dataset == 'pacbio_ENCFF450VAU':
                dataset_display = 'PacBio ENCFF450VAU'
            elif dataset == 'cDNA-NA12878':
                dataset_display = 'cDNA-NA12878'
            elif dataset == 'dRNA-NA12878':
                dataset_display = 'dRNA-NA12878'
            elif dataset == 'SRR307903':
                dataset_display = 'SRR307903'
            
            # Format assembler name
            assembler_display = assembler.replace('_', '\\_')
            if assembler == 'stringtie':
                assembler_display = 'StringTie'
            elif assembler == 'isoquant':
                assembler_display = 'IsoQuant'
            elif assembler == 'scallop2':
                assembler_display = 'Scallop2'
            
            row_parts = []
            
            # Handle dataset column with multirow
            if current_dataset != dataset:
                current_dataset = dataset
                dataset_row_count = dataset_counts[dataset]
                row_parts.append(f"\\multirow{{{dataset_row_count}}}{{*}}{{\\textbf{{{dataset_display}}}}}")
            else:
                row_parts.append("")
            
            # Add assembler and task
            row_parts.extend([assembler_display, task])
            
            # Add metric values
            for metric in available_metrics:
                val = row[metric]
                if isinstance(val, float):
                    row_parts.append(f"{val:.3f}")
                else:
                    row_parts.append(str(val))
            
            f.write(" & ".join(row_parts) + " \\\\\n")
            
            # Add midrule after each dataset group (except the last one)
            if idx < len(df_latex_filtered) - 1:
                next_dataset = df_latex_filtered.iloc[idx + 1]['dataset']
                if next_dataset != current_dataset:
                    f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table snippet saved to: {latex_file}")
    print(f"LaTeX table contains {len(df_latex_filtered)} XGBoost results with metrics: {available_metrics}")

def main():
    parser = argparse.ArgumentParser(
        description="Compile all metrics files into separate training and test tables for LaTeX"
    )
    parser.add_argument(
        '--output-prefix', '-o',
        default='compiled_metrics',
        help='Output file prefix (default: compiled_metrics). Will create _train.csv and _test.csv files'
    )
    parser.add_argument(
        '--config-dir',
        default='project_config',
        help='Directory containing config pickle files (default: project_config)'
    )
    
    args = parser.parse_args()
    
    # Get the project root directory (assuming script is in src/plotters)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Convert relative paths to absolute paths
    if os.path.isabs(args.config_dir):
        config_dir = args.config_dir
    else:
        config_dir = str(project_root / args.config_dir)
    
    if not os.path.exists(config_dir):
        print(f"Error: Config directory {config_dir} does not exist!")
        return
    
    print(f"Loading configs from: {config_dir}")
    
    # Load configs
    train_configs, test_configs = load_configs_from_directory(config_dir)
    
    print(f"Found {len(train_configs)} training configs and {len(test_configs)} test configs")
    
    if not train_configs and not test_configs:
        print("No valid configs found!")
        return
    
    # Compile training metrics
    train_df = pd.DataFrame()
    if train_configs:
        train_df = compile_metrics_from_configs(train_configs, 'train')
    
    # Compile test metrics
    test_df = pd.DataFrame()
    if test_configs:
        test_df = compile_metrics_from_configs(test_configs, 'test')
    
    # Determine output file paths
    if os.path.isabs(args.output_prefix):
        output_prefix = args.output_prefix
    else:
        output_prefix = str(project_root / args.output_prefix)
    
    train_output_file = f"{output_prefix}_train.csv"
    test_output_file = f"{output_prefix}_test.csv"
    
    # Save training results
    if not train_df.empty:
        format_for_latex(train_df, train_output_file)
        print(f"\nTraining results saved to: {train_output_file}")
        print(f"Training data shape: {train_df.shape}")
        print("\nSample of training data:")
        print(train_df.head())
        
        # Print training summary statistics
        print(f"\nTraining Summary:")
        print(f"- Datasets: {sorted(train_df['dataset'].unique())}")
        print(f"- Assemblers: {sorted(train_df['assembler'].unique())}")
        print(f"- Tasks: {sorted(train_df['task'].unique())}")
        print(f"- Models: {sorted(train_df['model'].unique())}")
    else:
        print("No training data found!")
    
    # Save test results
    if not test_df.empty:
        format_for_latex(test_df, test_output_file)
        print(f"\nTest results saved to: {test_output_file}")
        print(f"Test data shape: {test_df.shape}")
        print("\nSample of test data:")
        print(test_df.head())
        
        # Print test summary statistics
        print(f"\nTest Summary:")
        print(f"- Datasets: {sorted(test_df['dataset'].unique())}")
        print(f"- Assemblers: {sorted(test_df['assembler'].unique())}")
        print(f"- Tasks: {sorted(test_df['task'].unique())}")
        print(f"- Models: {sorted(test_df['model'].unique())}")
    else:
        print("No test data found!")

if __name__ == "__main__":
    main()
