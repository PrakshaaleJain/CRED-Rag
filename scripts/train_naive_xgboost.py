import os
import csv
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Credit Rating Mapping (21-point scale)
RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
}
REVERSE_RATING_MAP = {v: k for k, v in RATING_MAP.items()}
UNIQUE_CLASSES = sorted(list(set(RATING_MAP.values())))
CLASS_NAMES = [REVERSE_RATING_MAP[i] for i in UNIQUE_CLASSES]

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    hybrid_csv = project_root / 'data' / 'naive_training_features.csv'
    
    out_dir = project_root / 'output' / 'graphs' / 'xgboost'
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not hybrid_csv.exists():
        logging.error(f"{hybrid_csv} not found! Run scripts/build_naive_hybrid_dataset.py first.")
        return
        
    # 1. Load Data
    logging.info("Loading naive hybrid dataset...")
    df = pd.read_csv(hybrid_csv)
    
    # Sort chronologically for temporal split
    df = df.sort_values(by=['Year', 'CIK']).reset_index(drop=True)
    
    kpi_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    sent_features = [
        'Revenue_Sentiment', 'Operating Profit_Sentiment', 
        'Net/Gross Margins_Sentiment', 'Net Profit_Sentiment', 
        'Free Cash Flow_Sentiment'
    ]
    
    all_features = kpi_features + sent_features
    
    X = df[all_features].values
    y = df['Rating'].map(RATING_MAP).values
    
    # Drop rows with unmapped ratings if any
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    # 2. Imputation
    logging.info("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Drop ultra-rare classes with < 5 instances (SMOTE requires at least k_neighbors+1 instances)
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 5]
    valid_mask = np.isin(y, valid_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Re-encode labels to 0...N-1 to satisfy XGBoost requirement
    le = LabelEncoder()
    y = le.fit_transform(y)
    active_classes = list(range(len(le.classes_)))
    active_class_names = [REVERSE_RATING_MAP[orig] for orig in le.classes_]
    
    # 3. Train-Test Split (Temporal Split)
    logging.info("Performing chronological train-test split (80/20)...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Apply SMOTE to training data only
    logging.info("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # 4. Train Hybrid XGBoost
    logging.info("Training Naive RAG XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(active_classes),
        max_depth=5,                  # Allow deeper trees to capture Quant + Qual interactions
        n_estimators=500,             # High rounds (early stopping will halt naturally)
        learning_rate=0.1,
        subsample=0.9,                # Relaxed regularization
        colsample_bytree=0.9,
        min_child_weight=1,           # Back to default
        reg_lambda=1.0,               # Standard L2
        gamma=0.0,                    # Standard split threshold
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=30      # Increased patience
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    # Fit the model
    model.fit(
        X_train, y_train, 
        eval_set=eval_set,
        verbose=False
    )
    
    # Save Model
    model.save_model(models_dir / 'naive_xgb.json')
    
    # 5. Evaluation & Plots
    logging.info("Generating paper-ready plots...")
    y_pred = model.predict(X_test)
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=active_classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=active_class_names, yticklabels=active_class_names)
    plt.title('Naive RAG (Quant + Flat RAG) XGBoost Confusion Matrix', fontsize=16)
    plt.ylabel('True Rating', fontsize=12)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'naive_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Plot 2: Feature Importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(10, 8))
    
    # Color code features: Quant (skyblue) vs Qual (lightgreen)
    colors = ['lightgreen' if all_features[i] in sent_features else 'skyblue' for i in sorted_idx]
    
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color=colors)
    plt.yticks(range(len(sorted_idx)), [all_features[i] for i in sorted_idx])
    plt.title('Naive RAG Model Feature Importance', fontsize=14)
    plt.xlabel('XGBoost Relative Importance', fontsize=12)
    
    # Create custom legend for colors
    import matplotlib.patches as mpatches
    quant_patch = mpatches.Patch(color='skyblue', label='Quantitative KPIs')
    qual_patch = mpatches.Patch(color='lightgreen', label='Naive RAG Sentiment')
    plt.legend(handles=[quant_patch, qual_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'naive_feature_importance.png', dpi=300)
    plt.close()
    
    # Plot 3: Learning Curve
    results = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['mlogloss'], label='Train Log Loss', color='blue', linewidth=2)
    plt.plot(results['validation_1']['mlogloss'], label='Test Log Loss', color='orange', linewidth=2)
    plt.title('Naive RAG Training vs Validation Loss', fontsize=14)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Multi-Class Log Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / 'naive_learning_curve.png', dpi=300)
    plt.close()
    
    # Plot 4: Classification Report (CSV)
    report = classification_report(y_test, y_pred, labels=active_classes, target_names=active_class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / 'naive_classification_report.csv')
    
    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    # Calculate Ordinal Metrics
    mae = mean_absolute_error(y_test, y_pred)
    within_1_notch = np.mean(np.abs(y_test - y_pred) <= 1)
    
    logging.info(f"Naive RAG Accuracy:     {acc:.2%}")
    logging.info(f"Naive RAG Macro F1:     {macro_f1:.4f}  <-- (Preferred for imbalanced classes)")
    logging.info(f"Naive RAG Weighted F1:  {weighted_f1:.4f}")
    logging.info(f"Naive RAG MAE (Notches):{mae:.4f}")
    logging.info(f"Naive RAG Within-1-Notch: {within_1_notch:.2%}")
    
    logging.info(f"Plots saved to {out_dir}/")
    logging.info("Naive RAG training complete!")

if __name__ == '__main__':
    main()
