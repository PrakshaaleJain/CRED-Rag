import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from pathlib import Path

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
    hybrid_csv = project_root / 'data' / 'hybrid_training_features.csv'
    
    if not hybrid_csv.exists():
        logging.error(f"{hybrid_csv} not found! Run scripts/build_hybrid_dataset.py first.")
        return
        
    logging.info("Loading hybrid dataset...")
    df = pd.read_csv(hybrid_csv)
    
    # Sort chronologically
    df = df.sort_values(by=['Year', 'CIK']).reset_index(drop=True)
    
    # 1. Define Features
    quant_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    
    quant_delta_features = []
    for kpi in quant_features:
        delta_col = f'{kpi}_YoY_Change'
        df[delta_col] = df.groupby('CIK')[kpi].diff().fillna(0)
        quant_delta_features.append(delta_col)
        
    all_quant_features = quant_features + quant_delta_features
    
    sent_features = [
        'Revenue_Sentiment', 'Operating Profit_Sentiment', 
        'Net/Gross Margins_Sentiment', 'Net Profit_Sentiment', 
        'Free Cash Flow_Sentiment'
    ]
    
    all_features = all_quant_features + sent_features
    
    # Map target
    df['Bucket'] = df['Rating'].map(RATING_MAP)
    df = df.dropna(subset=['Bucket'])
    
    # Also drop rows where classes have < 2 samples to avoid GroupShuffleSplit and SMOTE errors
    class_counts = df['Bucket'].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df['Bucket'].isin(valid_classes)].reset_index(drop=True)
    
    groups = df['CIK'].values
    y_raw = df['Bucket'].values.astype(int)
    
    # Encode labels for XGBoost (must be 0 to num_class-1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    active_classes = list(range(len(le.classes_)))
    active_class_names = [REVERSE_RATING_MAP[orig] for orig in le.classes_]
    
    # 2. Impute Quant Features Globally (to prevent KNN failure)
    imputer = SimpleImputer(strategy='median')
    df[all_quant_features] = imputer.fit_transform(df[all_quant_features])
    
    # Also impute Sent Features Globally in case there are NaNs, before taking averages
    df[sent_features] = imputer.fit_transform(df[sent_features])
    
    # 3. Cold-Start Group Split (Isolate completely unseen CIKs)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups))
    
    df_universe = df.iloc[train_idx].copy()
    df_query = df.iloc[test_idx].copy()
    
    y_universe = y[train_idx]
    y_query = y[test_idx]
    
    logging.info(f"Peer Universe (Train): {len(df_universe)} companies")
    logging.info(f"Cold-Start Query (Test): {len(df_query)} companies")
    
    # 4. Train XGBoost Model on Universe
    # Extract universe arrays
    X_univ_train = df_universe[all_features].values
    
    # We must apply SMOTE to universe training to match our main methodology
    logging.info("Applying SMOTE to universe...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_univ_train_resampled, y_univ_train_resampled = smote.fit_resample(X_univ_train, y_universe)
    
    logging.info("Training Hybrid XGBoost on Universe...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(active_classes),
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=5.0,
        gamma=0.0,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_univ_train_resampled, y_univ_train_resampled, verbose=False)
    
    # 5. KNN Sentiment Proxy Imputation
    logging.info("Fitting KNN on Universe quantitative features...")
    scaler = StandardScaler()
    X_univ_quant = scaler.fit_transform(df_universe[all_quant_features].values)
    
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(X_univ_quant)
    
    logging.info("Proxying sentiment for cold-start companies...")
    X_query_quant = scaler.transform(df_query[all_quant_features].values)
    distances, indices = knn.kneighbors(X_query_quant)
    
    # For each query, average the sentiment of its 5 nearest neighbors
    imputed_sentiments = []
    for i in range(len(X_query_quant)):
        neighbor_idx = indices[i]
        # Get sentiments of neighbors
        neighbor_sents = df_universe.iloc[neighbor_idx][sent_features].values
        # Average them
        avg_sents = np.mean(neighbor_sents, axis=0)
        imputed_sentiments.append(avg_sents)
        
    imputed_sentiments = np.array(imputed_sentiments)
    
    # 6. Evaluate
    # Combine true quant features with IMPUTED qualitative features
    X_query_synthetic = np.hstack((df_query[all_quant_features].values, imputed_sentiments))
    
    y_pred = xgb_model.predict(X_query_synthetic)
    
    acc = accuracy_score(y_query, y_pred)
    macro_f1 = f1_score(y_query, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(y_query, y_pred)
    within_1_bucket = np.mean(np.abs(y_query - y_pred) <= 1)
    
    print("\n" + "="*50)
    print("=== KNN Sentiment Proxy Validation (Cold-Start) ===")
    print("="*50)
    print(f"Cold-Start Proxy Accuracy:      {acc:.2%}")
    print(f"Cold-Start Proxy Macro F1:      {macro_f1:.4f}")
    print(f"Cold-Start Proxy MAE (Buckets): {mae:.4f}")
    print(f"Cold-Start Proxy Within-1:      {within_1_bucket:.2%}\n")

if __name__ == "__main__":
    main()
