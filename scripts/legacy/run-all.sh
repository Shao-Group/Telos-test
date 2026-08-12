eval "$(conda shell.bash hook)"
conda activate berth

python src/train_all_data.py
python src/test_all.py

python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/
# python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/ --is_train
python src/plotters/generate_transcript_pr_curve.py --config_folder project_config/
python src/plotters/plot_venn.py --config_folder project_config/
python src/plotters/plot_venn.py --config_folder project_config/ --is_predictions --model_type xgboost
python src/plotters/plot_venn.py --config_folder project_config/ --is_predictions --model_type randomforest
python src/plotters/plot_venn_barplot.py 
# python src/plotters/calculate_true_false_stats.py
python src/plotters/gather_auc_results.py --barplot_two_tools
python src/plotters/plot_stage1_aupr_barplot.py --generate_all
# python src/plotters/feature_importance_plot.py --config_folder project_config/
# python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/ --is_train
# python src/plotters/generate_stage1_pr_curve.py --config_folder project_config/ --is_train

conda deactivate