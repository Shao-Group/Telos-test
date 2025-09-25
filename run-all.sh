eval "$(conda shell.bash hook)"
conda activate irtesam-berth

# python src/train_all_data.py
python src/test_all.py

python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/
python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/ --is_train
python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/
python src/plotters/feature_importance_plot.py --config_folder project_config/
python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/ --is_train
# python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/ --is_trai