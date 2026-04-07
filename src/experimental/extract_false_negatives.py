import os
import pandas as pd
import argparse

def get_start_ends(site_type, strand, position):
    if site_type.upper() == "TSS":
        if strand == "+":
            return position, position + 100
        elif strand == "-":
            return position - 100, position
    else:
        if strand == "+":
            return position - 100, position
        elif strand == "-":
            return position, position + 100
    return -1, -1

def extract_high_confidence_false_negatives(dataset_path, output_folder="false_negatives", 
                                           threshold=0.5, high_confidence_threshold=0.9):
    """
    Extract high confidence false negatives from a specific dataset.
    
    Args:
        dataset_path: Path to the dataset directory (e.g., test_output/dRNA-Hek293T_isoquant)
        output_folder: Output directory for false negative files
        threshold: Probability threshold for false negatives (sites with prob < threshold)
        high_confidence_threshold: Minimum probability for true positives to be considered "high confidence"
    """
    
    # Define file paths based on dataset structure
    features_dir = os.path.join(dataset_path, "features")
    predictions_dir = os.path.join(dataset_path, "predictions")
    
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path)
    
    # Process both TSS and TES
    for site_type in ["tss", "tes"]:
        print(f"\n=== Processing {site_type.upper()} for {dataset_name} ===")
        
        # File paths
        feature_file = os.path.join(features_dir, f"{dataset_name}_{site_type}_labeled.tsv")
        prediction_file = os.path.join(predictions_dir, f"{site_type}_xgboost_predictions.csv")
        
        if not os.path.exists(feature_file):
            print(f"Feature file not found: {feature_file}")
            continue
        if not os.path.exists(prediction_file):
            print(f"Prediction file not found: {prediction_file}")
            continue
            
        # Load data
        print(f"Loading features from: {feature_file}")
        print(f"Loading predictions from: {prediction_file}")
        
        features = pd.read_csv(feature_file, dtype={"chrom": str})
        preds = pd.read_csv(prediction_file, dtype={"chrom": str})
        
        print(f"Features shape: {features.shape}")
        print(f"Predictions shape: {preds.shape}")
        
        # Merge on chrom, position (strand info is in features)
        merged = pd.merge(
            features[["chrom", "position", "strand", "label"]],
            preds[["chrom", "position", "probability"]],
            on=["chrom", "position"],
            how="inner"
        )
        
        print(f"Merged data shape: {merged.shape}")
        
        # Identify different categories
        true_positives = merged[(merged["label"] == 1) & (merged["probability"] >= threshold)]
        false_negatives = merged[(merged["label"] == 1) & (merged["probability"] < threshold)]
        
        # High confidence false negatives: true sites that model is very confident are negative
        high_conf_false_negatives = false_negatives[false_negatives["probability"] < (1 - high_confidence_threshold)]
        
        print(f"\nSummary for {site_type.upper()}:")
        print(f"  Total positive sites: {len(merged[merged['label'] == 1])}")
        print(f"  True positives (prob >= {threshold}): {len(true_positives)}")
        print(f"  False negatives (prob < {threshold}): {len(false_negatives)}")
        print(f"  High confidence false negatives (prob < {1-high_confidence_threshold}): {len(high_conf_false_negatives)}")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Save high confidence false negatives
        output_bed = os.path.join(output_folder, f"{dataset_name}_{site_type}_high_conf_false_negatives.bed")
        output_csv = os.path.join(output_folder, f"{dataset_name}_{site_type}_high_conf_false_negatives.csv")
        
        # Save detailed CSV with all features
        high_conf_false_negatives_detailed = pd.merge(
            high_conf_false_negatives,
            features,
            on=["chrom", "position", "strand", "label"],
            how="left"
        )
        high_conf_false_negatives_detailed.to_csv(output_csv, index=False)
        
        # Write to BED format
        with open(output_bed, "w") as bed:
            bed.write("# High confidence false negatives for {} {}\n".format(dataset_name, site_type.upper()))
            bed.write("# chrom\tstart\tend\tname\tscore\tstrand\n")
            for _, row in high_conf_false_negatives.iterrows():
                chrom = row["chrom"]
                start, end = get_start_ends(site_type, row["strand"], row["position"])
                score = int(row["probability"] * 1000)  # Convert to integer score
                strand = row["strand"]
                name = f"{site_type.upper()}_FN_{row['position']}"
                bed.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")
        
        print(f"  High confidence false negatives saved to:")
        print(f"    BED: {output_bed}")
        print(f"    CSV: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract high confidence false negatives from dRNA datasets.")
    parser.add_argument('--dataset-path', default='test_output/dRNA-Hek293T_isoquant', 
                       help='Path to dataset directory (default: test_output/dRNA-Hek293T_isoquant)')
    parser.add_argument('--output-folder', default='false_negatives', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for false negatives')
    parser.add_argument('--high-confidence-threshold', type=float, default=0.75, 
                       help='High confidence threshold (FN with prob < 1-threshold)')

    args = parser.parse_args()
    
    # Call the main function
    extract_high_confidence_false_negatives(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        threshold=args.threshold,
        high_confidence_threshold=args.high_confidence_threshold
    ) 