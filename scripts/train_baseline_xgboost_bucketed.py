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

# Credit Rating Mapping (6-Bucket scale)
RATING_MAP = {
    'AAA': 5, 'AA+': 5, 'AA': 5, 'AA-': 5,
    'A+': 4, 'A': 4, 'A-': 4,
    'BBB+': 3, 'BBB': 3, 'BBB-': 3,
    'BB+': 2, 'BB': 2, 'BB-': 2,
    'B+': 1, 'B': 1, 'B-': 1,
    'CCC+': 0, 'CCC': 0, 'CCC-': 0, 'CC': 0, 'C': 0, 'D': 0
}
REVERSE_RATING_MAP = {
    5: 'AAA/AA',
    4: 'A',
    3: 'BBB',
    2: 'BB',
    1: 'B',
    0: 'CCC-D'
}

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
    
    # Sort chronologically for temporal split
    merged_df = merged_df.sort_values(by=['Year', 'CIK']).reset_index(drop=True)
    
    kpi_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    
    kpi_delta_features = []
    for kpi in kpi_features:
        delta_col = f'{kpi}_YoY_Change'
        merged_df[delta_col] = merged_df.groupby('CIK')[kpi].diff().fillna(0)
        kpi_delta_features.append(delta_col)
        
    all_features = kpi_features + kpi_delta_features
    
    X = merged_df[all_features].values
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
    
    # 4. Train Weak Baseline XGBoost
    logging.info("Training XGBoost baseline bucketed...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(active_classes),
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Save Model
    model.save_model(models_dir / 'baseline_xgb_bucketed.json')
    
    # 5. Evaluation & Plots
    logging.info("Generating paper-ready plots...")
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, labels=active_classes, target_names=active_class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / 'baseline_bucketed_classification_report.csv')
    
    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    # Color code features: Quant (skyblue) vs YoY Delta (plum)
    colors = []
    for i in sorted_idx:
        feat = all_features[i]
        if feat in kpi_delta_features:
            colors.append('plum')
        else:
            colors.append('skyblue')
            
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color=colors)
    plt.yticks(range(len(sorted_idx)), [all_features[i] for i in sorted_idx])
    plt.title('Baseline Bucketed Feature Importance', fontsize=14)
    plt.xlabel('XGBoost Relative Importance', fontsize=12)
    
    import matplotlib.patches as mpatches
    quant_patch = mpatches.Patch(color='skyblue', label='Quantitative KPIs')
    delta_patch = mpatches.Patch(color='plum', label='YoY Deltas')
    plt.legend(handles=[quant_patch, delta_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'baseline_bucketed_feature_importance.png', dpi=300)
    plt.close()
    
    weighted_f1 = report['weighted avg']['f1-score']
    
    mae = mean_absolute_error(y_test, y_pred)
    within_1_bucket = np.mean(np.abs(y_test - y_pred) <= 1)
    
    logging.info(f"Baseline Bucketed Accuracy:     {acc:.2%}")
    logging.info(f"Baseline Bucketed Macro F1:     {macro_f1:.4f}")
    logging.info(f"Baseline Bucketed Weighted F1:  {weighted_f1:.4f}")
    logging.info(f"Baseline Bucketed MAE (Buckets):{mae:.4f}")
    logging.info(f"Baseline Bucketed Within-1-Bucket: {within_1_bucket:.2%}")
    
    logging.info(f"Plots saved to {out_dir}/")
    logging.info("Baseline training complete!")

if __name__ == '__main__':
    main()
