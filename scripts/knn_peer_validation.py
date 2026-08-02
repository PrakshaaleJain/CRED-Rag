import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import os

RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
}

def load_and_merge_data(kpi_path, label_path):
    print("Loading data...")
    kpis = pd.read_csv(kpi_path)
    labels = pd.read_csv(label_path)
    
    kpis['CIK_Identifier'] = kpis['CIK_Identifier'].astype(str).str.zfill(10)
    labels['CIK'] = labels['CIK'].astype(str).str.zfill(10)
    
    merged = pd.merge(
        kpis, 
        labels, 
        left_on=['CIK_Identifier', 'Fiscal_Year'], 
        right_on=['CIK', 'Year'], 
        how='inner'
    )
    return merged

def run_knn_validation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kpi_path = os.path.join(base_dir, "data", "credit_risk_kpis_master.csv")
    label_path = os.path.join(base_dir, "data", "final_training_labels.csv")
    
    df = load_and_merge_data(kpi_path, label_path)
    
    if df.empty:
        print("Error: Merged dataframe is empty.")
        return
    
    quant_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    
    target_label = 'Rating'
    df = df.dropna(subset=[target_label])
    
    X = df[quant_features]
    # Map target to integers for ordinal metrics
    y = df[target_label].map(RATING_MAP).values
    groups = df['CIK'].values
    
    # Drop rows where rating didn't map
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    groups = groups[valid_idx]
    
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = pd.Series(y).isin(valid_classes)
    
    X = X[mask]
    y = y[mask]
    groups = groups[mask]
    
    # Split data grouped by CIK to prevent temporal leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_universe, X_query = X.iloc[train_idx], X.iloc[test_idx]
    y_universe, y_query = y[train_idx], y[test_idx]
    
    imputer = SimpleImputer(strategy='median')
    X_universe_imputed = imputer.fit_transform(X_universe)
    X_query_imputed = imputer.transform(X_query)
    
    scaler = StandardScaler()
    X_universe_scaled = scaler.fit_transform(X_universe_imputed)
    X_query_scaled = scaler.transform(X_query_imputed)
    
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(X_universe_scaled, y_universe)
    
    y_pred = knn.predict(X_query_scaled)
    
    accuracy = accuracy_score(y_query, y_pred)
    macro_f1 = f1_score(y_query, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(y_query, y_pred)
    within_1_notch = np.mean(np.abs(y_query - y_pred) <= 1)
    
    print("\n" + "="*50)
    print(f"=== KNN Peer Proxy Validation Results (K={k}) ===")
    print("="*50)
    print(f"Query Sample Size: {len(y_query)} companies")
    print(f"Peer Universe Size: {len(y_universe)} companies")
    print(f"KNN Baseline Accuracy:    {accuracy:.2%}")
    print(f"KNN Baseline Macro F1:    {macro_f1:.4f}")
    print(f"KNN Baseline MAE:         {mae:.4f} notches")
    print(f"KNN Baseline Within-1:    {within_1_notch:.2%}\n")

if __name__ == "__main__":
    run_knn_validation()
