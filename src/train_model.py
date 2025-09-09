import joblib
import argparse
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from config import Config, load_config
from ml_utils import stratified_split, evaluate_model, load_model, load_tmap_labels, chrom_to_int
from xgboost import XGBClassifier
from label_candidates import load_selected_features
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from multiprocessing import Pool, cpu_count
from functools import partial
from imblearn.under_sampling import RandomUnderSampler

def train_and_evaluate_stage2(df_tss, df_tes, project_config, model_type):
    df_cov = pd.read_csv(project_config.cov_file, sep="\t")
    df_label = load_tmap_labels(project_config.tmap_file)

    print(f"df_tss.shape: {df_tss.shape}")
    print(f"df_tes.shape: {df_tes.shape}")
    print(f"df_cov.shape: {df_cov.shape}")
    print(f"df_label.shape: {df_label.shape}")

    df = df_cov.merge(df_tss, left_on=["tss_chrom", "tss_pos"], right_on=["chrom", "position"], how="inner")
    df.drop(columns=["chrom", "position"], inplace=True)
    df = df.merge(df_tes, left_on=["tes_chrom", "tes_pos"], right_on=["chrom", "position"], how="inner", suffixes=("_tss", "_tes"))
    df = df.merge(df_label, left_on=["transcript_id"], right_on=["transcript_id"], how="inner")

    print(f"merged df.shape: {df.shape}")

    # MINIMAL MODIFICATION: Add transcript-level features before splitting
    df['transcript_length'] = np.abs(df['tes_pos'] - df['tss_pos'])
    df['log_transcript_length'] = np.log1p(df['transcript_length'])
    df['tss_confidence'] = df.get('probability_tss', 0.5)
    df['tes_confidence'] = df.get('probabilit   y_tes', 0.5) 
    df['min_confidence'] = np.minimum(df['tss_confidence'], df['tes_confidence'])
    df['confidence_product'] = df['tss_confidence'] * df['tes_confidence']
    
    X_train, X_test, y_train, y_test, train_mask, test_mask = stratified_split(df, validation_chrom_file=project_config.validation_chromosomes_file, train_chrom_file=project_config.train_chromosomes_file, return_mask=True)
    
    # MINIMAL MODIFICATION: Better feature selection for stage 2
    drop = [
        'chrom','position','chrom_tes','position_tes',
        'tss_chrom','tss_pos','tes_chrom','tes_pos',
        'site_type_tss','site_type_tes','strand', 'strand_tes',
        'ref_id','chrom_num','transcript_id','label', 
        'label_tss', 'label_tes'
    ]
    
    # Select base features + key TSS/TES features + transcript features
    base_features = [c for c in df.columns if c not in drop]
    features = [f for f in base_features if f in df.columns]  # Only keep existing features
    

    X_train = X_train[features]
    X_test  = X_test[features]

    print(f"Training stage 2 with {len(features)} features")
    # print(f"Key features:")
    # for f in features:
    #     print(f)
    
    print(f"Train transcript set shape stats:")
    print(f"Positive samples: {len(y_train[y_train==1])} --- {len(y_train[y_train==1]) / len(y_train)}")
    print(f"Negative samples: {len(y_train[y_train==0])} --- {len(y_train[y_train==0]) / len(y_train)}")
    
    print(f"Test transcript set shape stats:")
    print(f"Positive samples: {len(y_test[y_test==1])} --- {len(y_test[y_test==1]) / len(y_test)}")
    print(f"Negative samples: {len(y_test[y_test==0])} --- {len(y_test[y_test==0]) / len(y_test)}")
    
    # MINIMAL MODIFICATION: Optimize hyperparameters for transcript prediction
    clf_xgb = XGBClassifier(
            n_estimators=300,  # More trees
            max_depth=8,       # Deeper for interactions
            learning_rate=0.1, # Lower learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            # n_jobs=4,         # Use all CPU cores
            # tree_method='hist', # Faster training method
            # reg_lambda = 3,
            # reg_alpha = 0.5
            eval_metric='aucpr',
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)  # Handle imbalance
        )
    clf_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        random_state=42,
        # n_jobs=-1  # Use all CPU cores
        n_jobs=4
    )
    
    clf = Pipeline([
        # ('scaler', StandardScaler()),
        # ('clf', clf_rf)
        ('clf', clf_xgb)
    ])
    
    # Optional random undersampling on training data only (dynamic based on train set)
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    imbalance_ratio = (neg_count / max(pos_count, 1)) if pos_count > 0 else float('inf')
    if imbalance_ratio >= 2.0 and pos_count > 0:
        print(f"Applying RandomUnderSampler dynamically (majority/minority={imbalance_ratio:.2f}) to 1:1")
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print(f"Resampled train shape: {X_train.shape}, positives={np.sum(y_train==1)}, negatives={np.sum(y_train==0)}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Neural Network classifier
    # clf = TabularNeuralNetwork(random_state=42)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # y_prob = clf.predict_proba(X_test)[:, 1]

    # save model
    joblib.dump(clf, f"{project_config.models_output_dir}/xgboost_stage2_model.joblib")
    # clf.save_model(f"{project_config.models_output_dir}/neural_network_stage2_model.json")


    acc   = accuracy_score(y_test, y_pred)
    roc   = roc_auc_score(y_test, y_prob, average='macro')
    aupr  = average_precision_score(y_test, y_prob, average='macro')
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)

    print(f"Transcripts in test set: {len(y_test)}")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc:.4f}")
    print(f"AUPR:     {aupr:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("F1 score (macro):", report_dict['macro avg']['f1-score'])
    print("Precision (macro):", report_dict['macro avg']['precision'])
    print("Recall (macro):", report_dict['macro avg']['recall'])

    # save predictions
    pred_df = X_test.copy()
    pred_df['transcript_id'] = df.loc[test_mask, 'transcript_id'].values
    pred_df['pred_prob']      = y_prob
    pred_df['pred_label']     = y_pred
    pred_df['label']          = df.loc[test_mask, 'label'].values
    pred_df = pred_df[['transcript_id','pred_prob','pred_label','label']]

    pred_df.to_csv(os.path.join(project_config.predictions_output_dir, f"{model_type}_stage2_predictions_val.csv"),sep="\t" , index=False)


    y_pred_train = clf.predict(X_train)
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    pred_df_train = X_train.copy()
    pred_df_train['transcript_id'] = df.loc[train_mask, 'transcript_id'].values
    pred_df_train['pred_prob']      = y_prob_train
    pred_df_train['pred_label']     = y_pred_train
    pred_df_train['label']          = df.loc[train_mask, 'label'].values
    pred_df_train = pred_df_train[['transcript_id','pred_prob','pred_label','label']]
    pred_df_train.to_csv(os.path.join(project_config.predictions_output_dir, f"{model_type}_stage2_predictions_train.csv"),sep="\t" , index=False)

def train_single_model_stage1(args):
    """
    Wrapper function for parallel processing of stage 1 models.
    Takes a tuple of arguments to work with multiprocessing.Pool
    """
    df, model_type, model_config, project_config, site_type = args
    return train_and_evaluate_stage1(df, model_type, model_config, project_config, site_type)

def train_and_evaluate_stage1(df, model_type, model_config, project_config, site_type):
    drop = ["chrom", "position", "strand", "label"]

    X_train, X_val, y_train, y_val = stratified_split(df, validation_chrom_file=project_config.validation_chromosomes_file, train_chrom_file=project_config.train_chromosomes_file)
    
    # Use selected features if available, otherwise use all numeric features
    
    selected_features = load_selected_features(project_config, site_type)
    
    if selected_features:
        # Use only selected features that exist in the DataFrame
        available_selected_features = [col for col in selected_features if col in df.columns]
        print(f"Using {len(available_selected_features)} selected features for {site_type}")
        numeric_cols = available_selected_features
    else:
        # Fallback to all numeric features
        raise ValueError(f"No selected features found for {site_type}, using all numeric features")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in drop]
    
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]

    # Calculate class distribution for imbalance handling
    pos_count = len(y_train[y_train==1])
    neg_count = len(y_train[y_train==0])
    scale_pos_weight = neg_count / max(pos_count, 1)  # Avoid division by zero
    
    print(f"Class distribution for {site_type} {model_type}:")
    print(f"  Positive samples: {pos_count} ({pos_count/len(y_train)*100:.1f}%)")
    print(f"  Negative samples: {neg_count} ({neg_count/len(y_train)*100:.1f}%)")
    
    # Load model with appropriate imbalance handling
    if model_type == "xgboost":
        # Decide dynamically based on imbalance whether we will undersample
        will_undersample = (neg_count / max(pos_count, 1)) >= 2.0 and pos_count > 0
        if will_undersample:
            # When undersampling, avoid additional class weighting
            print("  Using undersampling; setting scale_pos_weight=1.0")
            model = load_model(model_type, model_config, scale_pos_weight=1.0)
        else:
            print(f"  XGBoost scale pos weight: {scale_pos_weight:.2f}")
            model = load_model(model_type, model_config, scale_pos_weight=scale_pos_weight)
    else:
        print(f"  Random Forest using class_weight='balanced'")
        model = load_model(model_type, model_config)

    # Optional random undersampling on training data only (dynamic based on train set)
    if (neg_count / max(pos_count, 1)) >= 2.0 and pos_count > 0:
        print(f"Applying RandomUnderSampler dynamically for {site_type} (majority/minority={(neg_count/max(pos_count,1)):.2f}) to 1:1")
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print(f"Resampled train shape: {X_train.shape}, positives={np.sum(y_train==1)}, negatives={np.sum(y_train==0)}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    
    if model_type == "xgboost":
        joblib.dump(model, os.path.join(project_config.models_output_dir, f"{site_type}_{model_type}_model.joblib"))
    elif model_type == "randomforest":
        joblib.dump(model, os.path.join(project_config.models_output_dir, f"{site_type}_{model_type}_model.joblib"))
    # elif model_type == "logistic":
    #     dump(model, os.path.join(project_config.models_output_dir, f"{site_type}_{model_type}_model.joblib"))

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pr_data_path = os.path.join(project_config.pr_data_dir, f"{site_type}_{model_type}_train_pr_data.csv")
    metrics = evaluate_model(y_train, y_train_pred, y_train_prob, y_train_pr_data_path )

    pr_data_path = os.path.join(project_config.pr_data_dir, f"{site_type}_{model_type}_val_pr_data.csv")
    print(f"Evaluating {site_type} {model_type} model")
    metrics = evaluate_model(y_val, y_pred, y_prob, pr_data_path )

    # save metrics to file
    metrics_list = []
    for metric_name, metric_value in metrics.items():
        if metric_name != 'confusion_matrix':  # Handle confusion matrix separately
            metrics_list.append({'metric': metric_name, 'value': metric_value})
    
    # Add confusion matrix as a formatted string
    cm_str = str(metrics['confusion_matrix'])
    metrics_list.append({'metric': 'confusion_matrix', 'value': cm_str})
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = os.path.join(project_config.metrics_output_dir, f"{site_type}_{model_type}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

    if model_type == "randomforest":
        importances = model.named_steps["clf"].feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_feats = np.array(numeric_cols)[sorted_idx]
        sorted_vals = importances[sorted_idx]

        # Save as CSV
        fi_df = pd.DataFrame({
            "feature": sorted_feats,
            "importance": sorted_vals
        })
        fi_df.to_csv(os.path.join(project_config.feature_importance_dir, f"{site_type}_rf_feature_importance.csv"), index=False)

    X_df = df.loc[:, numeric_cols].copy()
    y_prob = model.predict_proba(X_df)[:, 1]
    X_df["site_type"] = site_type.upper()
    X_df["chrom"] = df["chrom"].astype(str)
    X_df["position"] = df["position"].astype(int)
    X_df["strand"] = df["strand"].astype(str)
    X_df["probability"] = y_prob
    X_df = X_df[ ["site_type", "chrom", "position"] + numeric_cols + ["probability"] ]

    X_df.to_csv(os.path.join(project_config.predictions_output_dir, f"{site_type}_{model_type}_predictions.csv"), index=False)

    # pred_df_train = X_train.copy()

    
    return metrics, model, X_df

def train_tss_tes_parallel(df_tss, df_tes, model_type, model_config, project_config):
    """
    Train TSS and TES models in parallel using multiprocessing
    """
    print(f"🚀 Starting parallel training for {model_type} models (TSS and TES)")
    
    # Prepare arguments for parallel processing
    tss_args = (df_tss, model_type, model_config, project_config, "tss")
    tes_args = (df_tes, model_type, model_config, project_config, "tes")
    
    # Use multiprocessing to train both models simultaneously
    # Limit to 2 processes since we're only training 2 models
    num_processes = min(2, cpu_count())
    
    try:
        with Pool(processes=num_processes) as pool:
            # Submit both training tasks
            results = pool.map(train_single_model_stage1, [tss_args, tes_args])
        
        # Extract results
        (tss_metrics, tss_model, df_tss_pred) = results[0]
        (tes_metrics, tes_model, df_tes_pred) = results[1]
        
        print(f"✅ Parallel training completed for {model_type} models")
        print(f"   TSS model AUPR: {tss_metrics.get('aupr', 'N/A'):.4f}")
        print(f"   TES model AUPR: {tes_metrics.get('aupr', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"⚠️  Parallel training failed, falling back to sequential training: {e}")
        print("Training TSS model...")
        tss_metrics, tss_model, df_tss_pred = train_and_evaluate_stage1(df_tss, model_type, model_config, project_config, "tss")
        print("Training TES model...")
        tes_metrics, tes_model, df_tes_pred = train_and_evaluate_stage1(df_tes, model_type, model_config, project_config, "tes")
    
    return (tss_metrics, tss_model, df_tss_pred), (tes_metrics, tes_model, df_tes_pred)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--model-config-folder", required=True, help="Path to model config folder")
    parser.add_argument("--no-parallel", action="store_true", default=True, help="Disable parallel processing for TSS/TES models")
    args = parser.parse_args()

    project_config = load_config(args.project_config)
    project_config.validation_chromosomes_file = os.path.join(project_config.data_output_dir, "validation_chromosomes.txt")
    project_config.train_chromosomes_file = os.path.join(project_config.data_output_dir, "train_chromosomes.txt")

    for model_type in ["xgboost", "randomforest"]:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} models")
        print(f"{'='*60}")
        
        # Load datasets
        print(f"Loading datasets...")
        print(f"TSS data: {project_config.tss_labeled_file}")
        print(f"TES data: {project_config.tes_labeled_file}")
        
        df_tss = pd.read_csv(project_config.tss_labeled_file, dtype={"chrom": str})
        df_tes = pd.read_csv(project_config.tes_labeled_file, dtype={"chrom": str})
            
        with open(os.path.join(args.model_config_folder, f"{model_type}_config.yaml")) as f:
            model_config = yaml.safe_load(f)

        # Train TSS and TES models (parallel or sequential based on args)
        if args.no_parallel:
            print("🔄 Sequential training mode enabled")
            print("Training TSS model...")
            tss_metrics, tss_model, df_tss_pred = train_and_evaluate_stage1(df_tss, model_type, model_config, project_config, "tss")
            print("Training TES model...")
            tes_metrics, tes_model, df_tes_pred = train_and_evaluate_stage1(df_tes, model_type, model_config, project_config, "tes")
        else:
            (tss_metrics, tss_model, df_tss_pred), (tes_metrics, tes_model, df_tes_pred) = train_tss_tes_parallel(
                df_tss, df_tes, model_type, model_config, project_config
            )

        # Train Stage 2 model with the predictions from Stage 1
        print(f"\n🔄 Training Stage 2 model for {model_type}")
        train_and_evaluate_stage2(df_tss_pred, df_tes_pred, project_config, model_type)
    
    project_config.save_to_file(args.project_config)


if __name__ == "__main__":
    main()
    



# results for cdna from lrgasp
# results for all test/train
