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

parser = argparse.ArgumentParser(description="Extract false negative sites as BED and GTF.")
parser.add_argument('--feature-file', required=True, help='Path to labeled feature TSV file')
parser.add_argument('--prediction-file', required=True, help='Path to prediction CSV file')
parser.add_argument('--output-folder', default='false_negatives.gtf', help='Output GTF file')
parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for false negatives')

args = parser.parse_args()

# Read feature and prediction files
features = pd.read_csv(args.feature_file, sep=",")
preds = pd.read_csv(args.prediction_file, sep=",")

print("Prediction columns:", preds.columns.tolist())
print("Feature columns:", features.columns.tolist())

# Add strand info to preds (if not present, infer from features)
if "strand" not in preds.columns:
    preds = preds.merge(features[["chrom", "position", "strand"]], on=["chrom", "position"], how="left")

# Merge on chrom, position, strand
merged = pd.merge(
    features,
    preds[["chrom", "position", "strand", "probability"]],
    on=["chrom", "position", "strand"],
    how="inner"
)

basename_gtf = os.path.basename(args.prediction_file).split(".")[0]
basename_gtf = basename_gtf.replace("predictions", "false_negatives")
site_type = basename_gtf.split("_")[0]

# Identify false negatives
false_negatives = merged[(merged["label"] == 1) & (merged["probability"] < args.threshold)]

output_bed = os.path.join(args.output_folder, f"{basename_gtf}.bed")

# Create output directory if it doesn't exist
os.makedirs(args.output_folder, exist_ok=True)

# Write to BED
with open(output_bed, "w") as bed:
    for _, row in false_negatives.iterrows():
        chrom = row["chrom"]
        source = "FalseNegative"
        feature = site_type
        start, end = get_start_ends(site_type, row["strand"], row["position"])
        score = row["probability"]
        strand = row["strand"]
        frame = "."
        attributes = f'gene_id "FN"; transcript_id "FN";'
        bed.write(f"{chrom}\t{start}\t{end}\t{source}\t{score}\t{strand}\n")

print(f"False negatives written to {output_bed}")
print(f"Total false negatives found: {len(false_negatives)}") 