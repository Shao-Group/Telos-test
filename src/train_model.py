import argparse
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import StandardScaler
from config import Config, load_config
from ml_utils import stratified_split, evaluate_model, load_model, load_tmap_labels, chrom_to_int
from xgboost import XGBClassifier
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report
)

# class TabularNeuralNetwork:
#     """Neural Network wrapper for tabular data with sklearn-like interface"""
    
#     def __init__(self, random_state=42):
#         self.model = None
#         self.scaler = StandardScaler()
#         self.random_state = random_state
#         self.is_fitted = False
#         tf.random.set_seed(random_state)
#         np.random.seed(random_state)
    
#     def _build_model(self, input_dim):
#         """Build neural network architecture optimized for tabular data"""
#         model = Sequential([
#             # Input layer with batch normalization
#             Dense(128, activation='relu', input_shape=(input_dim,)),
#             BatchNormalization(),
#             Dropout(0.3),
            
#             # Hidden layers with decreasing size
#             Dense(96, activation='relu'),
#             BatchNormalization(), 
#             Dropout(0.25),
            
#             Dense(64, activation='relu'),
#             BatchNormalization(),
#             Dropout(0.2),
            
#             Dense(32, activation='relu'),
#             Dropout(0.15),
            
#             # Output layer for binary classification
#             Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(
#             optimizer=Adam(learning_rate=0.001),
#             loss='binary_crossentropy',
#             metrics=['accuracy', 'precision', 'recall']
#         )
        
#         return model
    
#     def fit(self, X, y):
#         """Fit the neural network"""
#         # Scale the features
#         X_scaled = self.scaler.fit_transform(X)
        
#         # Build model
#         self.model = self._build_model(X.shape[1])
        
#         # Callbacks for training
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=15,
#                 restore_best_weights=True,
#                 verbose=0
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=8,
#                 min_lr=1e-6,
#                 verbose=0
#             )
#         ]
        
#         # Train the model
#         history = self.model.fit(
#             X_scaled, y,
#             epochs=100,
#             batch_size=32,
#             validation_split=0.2,
#             callbacks=callbacks,
#             verbose=0
#         )
        
#         self.is_fitted = True
#         return self
    
#     def predict(self, X):
#         """Make binary predictions"""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before making predictions")
        
#         X_scaled = self.scaler.transform(X)
#         probabilities = self.model.predict(X_scaled, verbose=0)
#         return (probabilities > 0.5).astype(int).flatten()
    
#     def predict_proba(self, X):
#         """Predict class probabilities"""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before making predictions")
        
#         X_scaled = self.scaler.transform(X) 
#         probabilities = self.model.predict(X_scaled, verbose=0).flatten()
        
#         # Return probabilities in sklearn format [prob_class_0, prob_class_1]
#         return np.column_stack([1 - probabilities, probabilities])
    
#     def save_model(self, filepath):
#         """Save the model and scaler"""
#         if not self.is_fitted:
#             raise ValueError("Model must be fitted before saving")
        
#         # Save TensorFlow model
#         model_path = filepath.replace('.json', '_model.h5')
#         self.model.save(model_path)
        
#         # Save scaler
#         scaler_path = filepath.replace('.json', '_scaler.joblib')
#         dump(self.scaler, scaler_path)
        
#         # Save metadata
#         metadata = {
#             'model_path': model_path,
#             'scaler_path': scaler_path,
#             'random_state': self.random_state
#         }
        
#         import json
#         with open(filepath, 'w') as f:
#             json.dump(metadata, f)
    
#     def load_model(self, filepath):
#         """Load the model and scaler"""
#         import json
#         from joblib import load
        
#         # Load metadata
#         with open(filepath, 'r') as f:
#             metadata = json.load(f)
        
#         # Load TensorFlow model
#         self.model = tf.keras.models.load_model(metadata['model_path'])
        
#         # Load scaler
#         self.scaler = load(metadata['scaler_path'])
        
#         self.random_state = metadata['random_state']
#         self.is_fitted = True

def train_and_evaluate_stage2(df_tss, df_tes, project_config, model_type):
    df_cov = pd.read_csv(project_config.cov_file, sep="\t")
    df_label = load_tmap_labels(project_config.tmap_file)

    df = df_cov.merge(df_tss, left_on=["tss_chrom", "tss_pos"], right_on=["chrom", "position"], how="inner")
    df = df.merge(df_tes, left_on=["tes_chrom", "tes_pos"], right_on=["chrom", "position"], how="inner", suffixes=("", "_tes"))
    df = df.merge(df_label, left_on=["transcript_id"], right_on=["transcript_id"], how="inner")

    # df['chrom_num'] = df['tss_chrom'].apply(chrom_to_int)
    # train_mask = df['chrom_num'].between(1, 5)
    # X_train = df[train_mask]
    # X_test  = df[~train_mask]
    # y_train = X_train['label']
    # y_test  = X_test['label']
    X_train, X_test, y_train, y_test, train_mask, test_mask = stratified_split(df, validation_chrom_file=project_config.validation_chromosomes_file, return_mask=True)
    # select feature columns
    drop = [
        'chrom','position','chrom_tes','position_tes',
        'tss_chrom','tss_pos','tes_chrom','tes_pos',
        'site_type','site_type_tes',
        'ref_id','chrom_num','transcript_id','label'
    ]
    features = [c for c in df.columns
                if c not in drop and not c.startswith('tss_') and not c.startswith('tes_')]
    
    X_train = X_train[features]
    X_test  = X_test[features]

    print(f"Columns before feature selection: {df.columns}")
    print(f"Columns after feature selection: {features}")
    print(f"Training neural network with {len(features)} features")
    # print(f"Feature names: {features[:10]}..." if len(features) > 10 else f"Feature names: {features}")

    # XGBoost classifier (commented out)
    clf = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            eval_metric='aucpr'
        )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Neural Network classifier
    # clf = TabularNeuralNetwork(random_state=42)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # y_prob = clf.predict_proba(X_test)[:, 1]

    # save model
    clf.save_model(f"{project_config.models_output_dir}/xgboost_stage2_model.json")
    # clf.save_model(f"{project_config.models_output_dir}/neural_network_stage2_model.json")


    acc   = accuracy_score(y_test, y_pred)
    roc   = roc_auc_score(y_test, y_prob)
    aupr  = average_precision_score(y_test, y_prob)
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
    pred_df = pred_df[['transcript_id','pred_prob','pred_label']]

    pred_df.to_csv(os.path.join(project_config.predictions_output_dir, f"{model_type}_stage2_predictions.csv"),sep="\t" , index=False)


def train_and_evaluate_stage1(df, model_type, model_config, project_config, site_type):
    drop = ["chrom", "position", "strand", "label"]

    X_train, X_val, y_train, y_val = stratified_split(df, validation_chrom_file=project_config.validation_chromosomes_file)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in drop]
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]

    model = load_model(model_type, model_config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    
    if model_type == "xgboost":
        model.save_model(f"{project_config.models_output_dir}/{site_type}_{model_type}_model.json")
    elif model_type == "randomforest":
        dump(model, os.path.join(project_config.models_output_dir, f"{site_type}_{model_type}_model.joblib"))
    # elif model_type == "logistic":
    #     dump(model, os.path.join(project_config.models_output_dir, f"{site_type}_{model_type}_model.joblib"))

    pr_data_path = os.path.join(project_config.pr_data_dir, f"{site_type}_{model_type}_pr_data.csv")
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
    metrics_df.to_csv(os.path.join(project_config.metrics_output_dir, f"{site_type}_{model_type}_metrics.csv"), index=False)

    if model_type == "randomforest":
        importances = model.feature_importances_
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
    X_df["probability"] = y_prob
    X_df = X_df[ ["site_type", "chrom", "position"] + numeric_cols + ["probability"] ]

    X_df.to_csv(os.path.join(project_config.predictions_output_dir, f"{site_type}_{model_type}_predictions.csv"), index=False)

    
    return metrics, model, X_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--model-config-folder", required=True, help="Path to model config folder")
    args = parser.parse_args()

    project_config = load_config(args.project_config)
    project_config.validation_chromosomes_file = os.path.join(project_config.data_output_dir, "validation_chromosomes.txt")

    for model_type in ["xgboost", "randomforest"]:
        df_tss = pd.read_csv(project_config.tss_labeled_file)
        with open(os.path.join(args.model_config_folder, f"tss_{model_type}_config.yaml")) as f:
            model_config = yaml.safe_load(f)

        metrics, model, df_tss_pred = train_and_evaluate_stage1(df_tss, model_type, model_config, project_config, "tss")

        df_tes = pd.read_csv(project_config.tes_labeled_file)
        with open(os.path.join(args.model_config_folder, f"tes_{model_type}_config.yaml")) as f:
            model_config = yaml.safe_load(f)

        metrics, model, df_tes_pred = train_and_evaluate_stage1(df_tes, model_type, model_config, project_config, "tes")

        train_and_evaluate_stage2(df_tss_pred, df_tes_pred, project_config, model_type)
    
    project_config.save_to_file(args.project_config)


if __name__ == "__main__":
    main()
    

