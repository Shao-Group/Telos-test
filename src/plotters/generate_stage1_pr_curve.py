import os, sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from argparse import ArgumentParser
from config import load_config, Config


# —— USER SETTINGS ——
data_names = [
    "cDNA-NA12878","dRNA-ENCFF155CFF", "pacbio_ENCFF450VAU", "SRR307903"
]
name_dict = {
    "cDNA-NA12878" : "NA12878 cDNA",
    "dRNA-ENCFF155CFF" : "ENCFF155CFF dRNA",
    "pacbio_ENCFF450VAU": "ENCFF450VAU",
    "SRR307903" : "SRR307903"
}

# Define colors for different curve types
colors = {
    # Tool-specific colors
    "assembler1": "#000000",      # black
    "assembler2": "#808080",      # grey
    
    # Model-specific colors (add more as needed)
    "assembler1_xgboost": "#1f77b4", # blue
    "assembler1_randomforest": "#008000", # green
    "assembler2_xgboost": "#ff7f0e", # orange
    "assembler2_randomforest": "#ff0000" # red
}

STAGE1_PR_PLOT_FOLDER = "plots/"

# ———————

# create a 4×2 grid: 4 datasets × 2 site-types
def plot_pr(config_folder):
    site_types = ['tss', 'tes']
    fig, axes = plt.subplots(
        nrows=len(data_names),
        ncols=2,
        figsize=(12, 3 * len(data_names)),
        sharex=False,
        sharey=False
    )
    
    # Track legend elements
    legend_elements = []
    legend_labels_set = set()  # To avoid duplicates
    legend_labels = []

    for i, name in enumerate(data_names):
        tools =  ["stringtie", "scallop2"] if name.startswith("SRR") else ["stringtie", "isoquant"]
        tool_map = {
            "stringtie" : "assembler1",
            "scallop2" : "assembler2",
            "isoquant" : "assembler2"
        }

        configs  = {
            tools[0] : load_config(os.path.join(config_folder, f"{name}_{tools[0]}_config.pkl")),
            tools[1] : load_config(os.path.join(config_folder, f"{name}_{tools[1]}_config.pkl"))
        }
        for j, site in enumerate(site_types):
            print(f"Plotting {name} {site} {i} {j}")
            ax = axes[i, j]
            
            pr_files  = {
                tools[0] : os.listdir(configs[tools[0]].pr_data_dir), 
                tools[1] : os.listdir(configs[tools[1]].pr_data_dir)
            }
            
            
            for tool, files in pr_files.items():
                for fname in files:
                    assert fname.endswith("_pr_data.csv")
                    if site not in fname:
                        continue

                    site_model = fname.replace("_pr_data.csv", "")
                    # if  "randomforest" in site_model :
                    #     continue
                    pr_df = pd.read_csv(os.path.join(configs[tool].pr_data_dir, fname))

                    model_type = fname.split("_")[1]  # e.g., "logistic"
                    label = f"{tool.title()} {model_type.title()}"
                    color_key = f"{tool_map[tool]}_{model_type}"
                    color = colors.get(color_key, colors.get(tool, "#333333"))
                    
                    # full PR curve
                    line = ax.plot(
                        pr_df["recall"],
                        pr_df["precision"],
                        label=label,
                        color=color,
                        linewidth=2
                    )[0]
                    
                    # Add to legend if not already present
                    # legend_label = f"{tool_map[tool]}"
                    if color_key not in legend_labels_set:
                        legend_elements.append(line)
                        legend_labels_set.add(color_key)
                        legend_labels.append(color_key.replace("_", " ").title())

                # baseline: coverage-based PR curve
                labeled_data_file = configs[tool].tss_labeled_file if site == "tss" else configs[tool].tes_labeled_file
                baseline_df = pd.read_csv(labeled_data_file, dtype={"chrom": str, "position": int})
                # print(f"Number of original points: {len(baseline_df)} for {tool} {site}")

                # print([c for c in baseline_df.columns])
                df_cov = pd.read_csv(configs[tool].cov_file, sep="\t")
                # print(f"Coverage file shape: {df_cov.shape}")

                df_cov = df_cov[[f"{site}_chrom", f"{site}_pos", "coverage"]]
                
                # Check for duplicates in coverage file
                n_unique_positions = df_cov[[f"{site}_chrom", f"{site}_pos"]].drop_duplicates().shape[0]
                print(f"Unique positions in coverage file: {n_unique_positions}, Total rows: {len(df_cov)} for {tool} {site}")
                
                if len(df_cov) > n_unique_positions:
                    # print(f"WARNING: Coverage file has duplicates! Averaging coverage values...")
                    # Average coverage for duplicate positions
                    df_cov = df_cov.groupby([f"{site}_chrom", f"{site}_pos"])['coverage'].mean().reset_index()
                
                baseline_df = baseline_df.merge(df_cov, left_on=["chrom", "position"], right_on=[f"{site}_chrom", f"{site}_pos"], how="inner")
                # print(f"Number of filtered points: {len(baseline_df)} for {tool} {site}")
                # # print sample missing points
                # if baseline_df[baseline_df["coverage"].isna()].shape[0] > 0:
                #     print(baseline_df[baseline_df["coverage"].isna()].sample(min(10, baseline_df[baseline_df["coverage"].isna()].shape[0])))

                with open (configs[tool].validation_chromosomes_file, "r") as fp:
                    val_chrom = [line.rstrip("\n") for line in fp]

                # Filter to validation chromosomes and sort by coverage
                baseline_filtered = baseline_df.loc[baseline_df["chrom"].isin(val_chrom)].copy()
                baseline_filtered = baseline_filtered.sort_values('coverage', ascending=False)
                
                # Generate PR curve by gradually removing low coverage points
                precisions = []
                recalls = []
                n_total = len(baseline_filtered)
                n_positive = baseline_filtered['label'].sum()
                
                for k in range(1, n_total + 1):
                    # Keep top i highest coverage points
                    subset = baseline_filtered.iloc[:k]
                    n_kept = len(subset)
                    n_tp = subset['label'].sum()  # true positives in kept subset
                    n_fp = n_kept - n_tp  # false positives in kept subset
                    
                    # Precision = TP / (TP + FP)
                    precision = n_tp / n_kept 
                    # Recall = TP / (total positive)  
                    recall = n_tp / n_positive 
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Plot coverage-based baseline PR curve
                baseline_label = f"{tool_map[tool].title()} Baseline"
                baseline_color = colors.get(f"{tool_map[tool]}", colors.get(tool, "#333333"))
                
                baseline_line = ax.plot(
                    recalls,
                    precisions,
                    label=baseline_label,
                    # linestyle="dotted",
                    linewidth=2,
                    color=baseline_color,
                    alpha=0.8
                )[0]

                # add a point at the end of the curve
                ax.plot(
                    [recalls[-1]],
                    [precisions[-1]],
                    marker="o",
                    color=baseline_color,
                    markersize=5,
                    alpha=0.5
                )
                
                # Add to legend if not already present
                if baseline_label not in legend_labels:
                    legend_elements.append(baseline_line)
                    legend_labels_set.add(baseline_label)
                    legend_labels.append(baseline_label)

            # formatting per‐subplot
            ax.set_title(f"{name_dict[name]} - {site.upper()}", fontsize=12)
            if i == len(data_names) - 1:
                ax.set_xlabel("Recall", fontsize=10)
            if j == 0:
                ax.set_ylabel("Precision", fontsize=10)
            # ax.tick_params(labelsize= nine)  # increase tick labels if you wish
            # Remove individual subplot legends
            if ax.get_legend():
                ax.legend().remove()

    # legend_labels = list(set(legend_labels))

    # Create a single legend at the top of the figure
    if legend_elements:
        fig.legend(
            legend_elements, 
            # [element.get_label() for element in legend_elements],
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),  # Position at top center
            ncol=min(len(legend_elements), 3),  # Max 4 columns
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True
        )
    
    # Alternative: Manual legend with custom elements (uncomment if needed)
    # manual_legend_elements = [
    #     mlines.Line2D([0], [0], color='#1f77b4', linewidth=2, label='Stringtie Logistic'),
    #     mlines.Line2D([0], [0], color='#1f77b4', linewidth=2, linestyle='--', label='Stringtie Baseline'),
    #     mlines.Line2D([0], [0], color='#ff7f0e', linewidth=2, label='Scallop2 Logistic'),
    #     mlines.Line2D([0], [0], color='#ff7f0e', linewidth=2, linestyle='--', label='Scallop2 Baseline'),
    #     mlines.Line2D([0], [0], color='#2ca02c', linewidth=2, label='Isoquant Logistic'),
    #     mlines.Line2D([0], [0], color='#2ca02c', linewidth=2, linestyle='--', label='Isoquant Baseline'),
    # ]
    # fig.legend(handles=manual_legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10)
        
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(top=0.89)
    plt.savefig(os.path.join(STAGE1_PR_PLOT_FOLDER,"all_pr_curves_grid.pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {STAGE1_PR_PLOT_FOLDER}/all_pr_curves_grid.pdf")


def main():
    os.makedirs(STAGE1_PR_PLOT_FOLDER, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    plot_pr(args.config_folder)

if __name__ == "__main__":
    main()