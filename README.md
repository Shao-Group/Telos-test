# Telos-test

This repository is dedicated for the experimentation of Telos. You should have all the packages required for Telos installed. Additionally, you must have [rnaseqtools](https://github.com/Shao-Group/rnaseqtools) and [GFFCompare](https://anaconda.org/bioconda/gffcompare) installed. 

## Quickstart

You can reproduce the full training, testing, and plotting workflow with the provided script:

```bash
./run-all.sh
```

Notes:
- The script activates a Conda environment named `berth`. Adjust if your environment name differs.
- Update any paths in your config as needed. The default config folder used is `project_config/`.

## Manual usage

Train on all training data:

```bash
python src/train_all_data.py
```

Test on evaluation datasets:

```bash
python src/test_all.py
```
For both of this scripts, update the paths as necessary.

Generate plots and summaries (aligns with `run-all.sh`):

- Stage 1 PR curve

  ```bash
  python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/
  ```

- Transcript-level PR curve (optionally add `--is_train` to use training data)

  ```bash
  python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/
  # python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/ --is_train
  ```

- Venn diagram of methods (ground truth)

  ```bash
  python src/plotters/plot_venn.py --config_folder project_config/
  ```

- Venn diagram of predictions (per model)

  ```bash
  python src/plotters/plot_venn.py --config_folder project_config/ --is_predictions --model_type xgboost
  python src/plotters/plot_venn.py --config_folder project_config/ --is_predictions --model_type randomforest
  ```

- For the Jaccard Similarity bar plot

  ```bash
  python src/plotters/plot_venn_barplot.py
  ```

- Aggregate transcript level AUPR results across runs

  ```bash
  python src/plotters/gather_auc_results.py --barplot_two_tools
  ```

- Plot Stage 1 AuPR Barplots

  ```bash
  python src/plotters/plot_stage1_aupr_barplot.py --generate_all
  ```

<!-- 
Optional (commented in `run-all.sh`):

```bash
# python src/plotters/feature_importance_plot.py --config_folder project_config/
# python src/plotters/calculate_true_false_stats.py
# python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/ --is_train
``` -->


---

## ✍️ Author

Developed by [Shao Group](https://github.com/Shao-Group) .


---

For help or issues, open an issue on GitHub or contact the author.
