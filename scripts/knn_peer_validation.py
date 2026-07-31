import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import os

def load_and_merge_data(kpi_path, label_path):
    print("Loading data...")
    # Load KPIs
    kpis = pd.read_csv(kpi_path)
    # Load Labels
    labels = pd.read_csv(label_path)
    
    print(f"Original KPIs shape: {kpis.shape}")
    print(f"Original Labels shape: {labels.shape}")
    
    # Ensure CIKs are standard strings to merge properly
    kpis['CIK_Identifier'] = kpis['CIK_Identifier'].astype(str).str.zfill(10)
    labels['CIK'] = labels['CIK'].astype(str).str.zfill(10)
    
    # Merge on CIK and Year
    merged = pd.merge(
        kpis, 
        labels, 
        left_on=['CIK_Identifier', 'Fiscal_Year'], 
        right_on=['CIK', 'Year'], 
        how='inner'
    )
    
    print(f"Merged shape: {merged.shape}")
    return merged

def run_knn_validation():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kpi_path = os.path.join(base_dir, "data", "credit_risk_kpis_master.csv")
    label_path = os.path.join(base_dir, "data", "final_training_labels.csv")
    
    df = load_and_merge_data(kpi_path, label_path)
    
    if df.empty:
        print("Error: Merged dataframe is empty. Please check CIK and Year formats.")
        return
    
    # Define quantitative features
    quant_features = [
        'Debt-to-Equity', 
        'Retained Earnings / Total Assets', 
        'Current Ratio', 
        'Quick Ratio', 
        'Working Capital / Total Assets', 
        'ROCE', 
        'Net Profit Margin', 
        'Total Liabilities / Total Assets'
    ]
    
    target_label = 'Rating'
    
    # Drop rows where the target label is missing
    df = df.dropna(subset=[target_label])
    print(f"Shape after dropping missing labels: {df.shape}")
    
    # We will use simple imputation for missing quant features to preserve data
    X = df[quant_features]
    y = df[target_label]
    
    # Filter classes with very few samples to allow stratification
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    print(f"Shape after removing rare classes (for stratification): {X.shape}")
    
    # Split data (15% query set, 85% peer universe)
    X_universe, X_query, y_universe, y_query = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Impute missing values (using median of the training/universe set)
    imputer = SimpleImputer(strategy='median')
    X_universe_imputed = imputer.fit_transform(X_universe)
    X_query_imputed = imputer.transform(X_query)
    
    # Scale Features
    scaler = StandardScaler()
    X_universe_scaled = scaler.fit_transform(X_universe_imputed)
    X_query_scaled = scaler.transform(X_query_imputed)
    
    # Fit KNN (K=5)
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(X_universe_scaled, y_universe)
    
    # Predict
    y_pred = knn.predict(X_query_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_query, y_pred)
    
    print("\n" + "="*50)
    print(f"=== KNN Peer Proxy Validation Results (K={k}) ===")
    print("="*50)
    print(f"Query Sample Size: {len(y_query)} companies")
    print(f"Peer Universe Size: {len(y_universe)} companies")
    print(f"Baseline Accuracy (Quants Only): {accuracy:.2%}\n")
    
    print("Detailed Classification Report:")
    # We only show labels that were in the test set to keep the report clean
    print(classification_report(y_query, y_pred, zero_division=0))

if __name__ == "__main__":
    run_knn_validation()
