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
from sklearn.ensemble import RandomForestClassifier
import joblib

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
    
    hybrid_csv = project_root / 'data' / 'hybrid_training_features.csv'
    
    out_dir = project_root / 'output' / 'graphs'
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not hybrid_csv.exists():
        logging.error(f"{hybrid_csv} not found! Run scripts/build_hybrid_dataset.py first.")
        return
        
    # 1. Load Data
    logging.info("Loading hybrid dataset...")
    df = pd.read_csv(hybrid_csv)
    
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
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Hybrid Random Forest
    logging.info("Training Hybrid Random Forest model (same configuration as baseline)...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save Model
    joblib.dump(model, models_dir / 'hybrid_rf.joblib')
    
    # 5. Evaluation & Plots
    logging.info("Generating paper-ready plots...")
    y_pred = model.predict(X_test)
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=UNIQUE_CLASSES)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Hybrid (Quant + Qual) Random Forest Confusion Matrix', fontsize=16)
    plt.ylabel('True Rating', fontsize=12)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'hybrid_rf_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Plot 2: Feature Importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(10, 8))
    
    # Color code features: Quant (skyblue) vs Qual (lightgreen)
    colors = ['lightgreen' if all_features[i] in sent_features else 'skyblue' for i in sorted_idx]
    
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color=colors)
    plt.yticks(range(len(sorted_idx)), [all_features[i] for i in sorted_idx])
    plt.title('Hybrid Model Feature Importance (Random Forest)', fontsize=14)
    plt.xlabel('Gini Importance', fontsize=12)
    
    # Create custom legend for colors
    import matplotlib.patches as mpatches
    quant_patch = mpatches.Patch(color='skyblue', label='Quantitative KPIs')
    qual_patch = mpatches.Patch(color='lightgreen', label='Qualitative Sentiment')
    plt.legend(handles=[quant_patch, qual_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'hybrid_rf_feature_importance.png', dpi=300)
    plt.close()
    
    # Plot 3: Classification Report (CSV)
    report = classification_report(y_test, y_pred, labels=UNIQUE_CLASSES, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / 'hybrid_rf_classification_report.csv')
    
    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    logging.info(f"Hybrid RF Accuracy:     {acc:.2%}")
    logging.info(f"Hybrid RF Macro F1:     {macro_f1:.4f}  <-- (Preferred for imbalanced classes)")
    logging.info(f"Hybrid RF Weighted F1:  {weighted_f1:.4f}")
    
    logging.info(f"Plots saved to {out_dir}/")
    logging.info("Hybrid training complete!")

if __name__ == '__main__':
    main()
