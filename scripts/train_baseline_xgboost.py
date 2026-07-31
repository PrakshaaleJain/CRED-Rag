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
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Credit Rating Mapping (21-point scale)
RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
} # C and D merged to 0 for simplicity if 21 classes, but let's keep it strictly mapped
# Wait, AAA is 21 classes if we have AAA to D. Let's make it 21 distinct classes 0-20.
RATING_MAP_21 = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0
}
# Some datasets have 'D' as a default, we will map 'D' to 0 as well to be safe.
RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
}
REVERSE_RATING_MAP = {v: k for k, v in RATING_MAP.items()}
# To ensure unique reverse mapping for labels in plots
UNIQUE_CLASSES = sorted(list(set(RATING_MAP.values())))
CLASS_NAMES = [REVERSE_RATING_MAP[i] for i in UNIQUE_CLASSES]

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    kpis_csv = project_root / 'data' / 'credit_risk_kpis_master.csv'
    rated_csv = project_root / 'data' / 'final_training_labels.csv'
    
    out_dir = project_root / 'output' / 'graphs' / 'xgboost'
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logging.info("Loading datasets...")
    kpis_df = pd.read_csv(kpis_csv)
    kpis_df['CIK_Identifier'] = kpis_df['CIK_Identifier'].astype(str).str.zfill(10)
    kpis_df['Fiscal_Year'] = kpis_df['Fiscal_Year'].astype(str)
    
    rated_df = pd.read_csv(rated_csv)
    rated_df['CIK'] = rated_df['CIK'].astype(str).str.zfill(10)
    rated_df['Year'] = rated_df['Year'].astype(str)
    
    # Filter rated companies
    rated_df = rated_df[(rated_df['Rating'] != 'NR')]
    if 'SOURCE_INDICATOR' in rated_df.columns:
        rated_df = rated_df[rated_df['SOURCE_INDICATOR'].astype(str) != '0']
        
    merged_df = pd.merge(rated_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    
    kpi_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    
    X = merged_df[kpi_features].values
    y = merged_df['Rating'].map(RATING_MAP).values
    
    # Drop rows with unmapped ratings if any
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    # 2. Imputation
    logging.info("Imputing missing KPI values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Drop ultra-rare classes with < 5 instances (SMOTE requires at least k_neighbors+1 instances)
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 5]
    valid_mask = np.isin(y, valid_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to training data only
    logging.info("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # 4. Train Weak Baseline XGBoost
    logging.info("Training XGBoost baseline (max_depth=4, n_estimators=150)...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(UNIQUE_CLASSES),
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Save Model
    model.save_model(models_dir / 'baseline_xgb.json')
    
    # 5. Evaluation & Plots
    logging.info("Generating paper-ready plots...")
    y_pred = model.predict(X_test)
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=UNIQUE_CLASSES)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Baseline Quant-Only XGBoost Confusion Matrix', fontsize=16)
    plt.ylabel('True Rating', fontsize=12)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'baseline_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Plot 2: Feature Importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color='skyblue')
    plt.yticks(range(len(sorted_idx)), [kpi_features[i] for i in sorted_idx])
    plt.title('Baseline KPI Feature Importance', fontsize=14)
    plt.xlabel('XGBoost Relative Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'baseline_feature_importance.png', dpi=300)
    plt.close()
    
    # Plot 3: Learning Curve
    results = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['mlogloss'], label='Train Log Loss', color='blue', linewidth=2)
    plt.plot(results['validation_1']['mlogloss'], label='Test Log Loss', color='orange', linewidth=2)
    plt.title('Baseline Training vs Validation Loss (Restricted Boosting)', fontsize=14)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Multi-Class Log Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / 'baseline_learning_curve.png', dpi=300)
    plt.close()
    
    # Plot 4: Classification Report (CSV)
    report = classification_report(y_test, y_pred, labels=UNIQUE_CLASSES, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / 'baseline_classification_report.csv')
    
    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    logging.info(f"Baseline Accuracy:     {acc:.2%}")
    logging.info(f"Baseline Macro F1:     {macro_f1:.4f}  <-- (Preferred for imbalanced classes)")
    logging.info(f"Baseline Weighted F1:  {weighted_f1:.4f}")
    
    logging.info(f"Plots saved to {out_dir}/")
    logging.info("Baseline training complete!")

if __name__ == '__main__':
    main()
