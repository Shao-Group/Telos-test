# Telos-test

This repository is dedicated for the experimentation of Telos. You should have all the packages required for Telos installed. Additionally, you must have `rnaseqtools` and `GFFCompare` installed. 

Run src/train_all_data.py for training all the training dataset.
Run src/test_all.py for testing on different datasets.

For both cases, update the paths as necessary. To generate all plots presented in the manuscript, run the four scripts inside `src/plotters/`.

 - `python src/plotters/plot_venn.py --config_folder CONFIG_FOLDER`
 - `python src/plotters/generate_stage1_pr_curve.py --config_folder CONFIG_FOLDER`
 - `python src/plotters/generate_transcript_pr_curve.py --config_folder CONFIG_FOLDER`
 - `python src/plotters/feature_importance_plot.py --config_folder CONFIG_FOLDER`


---

## ✍️ Author

Developed by [Shao Lab](https://github.com/Shao-Group) .


---

For help or issues, open an issue on GitHub or contact the author.
