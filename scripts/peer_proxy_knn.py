import os
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

# Credit Rating Mapping (21-point scale)
RATING_MAP = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1, 'D': 0
}
REVERSE_RATING_MAP = {v: k for k, v in RATING_MAP.items()}

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    
    kpis_csv = project_root / 'data' / 'credit_risk_kpis_master.csv'
    rated_csv = project_root / 'data' / 'final_training_labels.csv'
    unrated_csv = project_root / 'data' / 'unrated_inference_set.csv'
    out_csv = project_root / 'data' / 'knn_proxy_predictions.csv'
    
    if not kpis_csv.exists():
        logging.error(f"{kpis_csv} not found!")
        return

    # 1. Load KPIs
    logging.info("Loading quantitative KPIs...")
    kpis_df = pd.read_csv(kpis_csv)
    kpis_df['CIK_Identifier'] = kpis_df['CIK_Identifier'].astype(str).str.zfill(10)
    kpis_df['Fiscal_Year'] = kpis_df['Fiscal_Year'].astype(str)
    
    kpi_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    
    # Preprocess KPIs (Impute and Scale)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # We fit imputer on ALL data to ensure consistency
    kpis_df[kpi_features] = imputer.fit_transform(kpis_df[kpi_features])
    
    # 2. Load Knowledge Base (Rated Companies)
    logging.info("Building Knowledge Base of rated companies...")
    rated_df = pd.read_csv(rated_csv)
    rated_df['CIK'] = rated_df['CIK'].astype(str).str.zfill(10)
    rated_df['Year'] = rated_df['Year'].astype(str)
    
    # Filter only valid rated companies
    rated_df = rated_df[(rated_df['Rating'] != 'NR')]
    if 'SOURCE_INDICATOR' in rated_df.columns:
        rated_df = rated_df[rated_df['SOURCE_INDICATOR'].astype(str) != '0']
        
    kb_merged = pd.merge(rated_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    
    # Map ratings to numerical scale
    kb_merged['Rating_Score'] = kb_merged['Rating'].map(lambda x: RATING_MAP.get(x, 10))
    
    # Extract features array and scale
    kb_features = kb_merged[kpi_features].values
    kb_features_scaled = scaler.fit_transform(kb_features) # Fit scaler on KB
    
    # 3. Process Inference Set (Unrated Companies)
    logging.info("Processing unrated companies...")
    unrated_df = pd.read_csv(unrated_csv)
    unrated_df['CIK'] = unrated_df['CIK'].astype(str).str.zfill(10)
    unrated_df['Year'] = unrated_df['Year'].astype(str)
    
    inf_merged = pd.merge(unrated_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    
    if inf_merged.empty:
        logging.error("No unrated companies matched with KPI data.")
        return
        
    inf_features = inf_merged[kpi_features].values
    inf_features_scaled = scaler.transform(inf_features)
    
    # 4. Perform KNN on Quantitative Features (Euclidean distance)
    logging.info("Calculating K-Nearest Neighbors (K=5) using quantitative features...")
    K = 5
    nn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    nn.fit(kb_features_scaled)
    
    distances, indices = nn.kneighbors(inf_features_scaled)
    
    results = []
    
    for i in range(len(inf_merged)):
        inf_cik = inf_merged.iloc[i]['CIK']
        inf_year = inf_merged.iloc[i]['Year']
        
        top_k_idx = indices[i]
        sim_scores = distances[i] # These are distances, so lower is better
        
        peer_ratings = [kb_merged.iloc[idx]['Rating_Score'] for idx in top_k_idx]
        avg_rating_score = round(float(np.mean(peer_ratings)))
        predicted_rating = REVERSE_RATING_MAP.get(avg_rating_score, 'NR')
        
        row_data = [inf_cik, inf_year, predicted_rating, avg_rating_score]
        
        # Add peer details
        for rank, idx in enumerate(top_k_idx):
            peer_cik = kb_merged.iloc[idx]['CIK']
            peer_rating = kb_merged.iloc[idx]['Rating']
            peer_dist = round(float(sim_scores[rank]), 4)
            row_data.extend([peer_cik, peer_rating, peer_dist])
            
        results.append(row_data)
        
    # 5. Export
    logging.info(f"Exporting predictions to {out_csv}...")
    headers = [
        "CIK", "Year", "Predicted_Rating", "Avg_Rating_Score",
        "Peer1_CIK", "Peer1_Rating", "Peer1_Distance",
        "Peer2_CIK", "Peer2_Rating", "Peer2_Distance",
        "Peer3_CIK", "Peer3_Rating", "Peer3_Distance",
        "Peer4_CIK", "Peer4_Rating", "Peer4_Distance",
        "Peer5_CIK", "Peer5_Rating", "Peer5_Distance"
    ]
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
        
    logging.info("Done! Peer proxies calculated using Quantitative KPIs.")

if __name__ == '__main__':
    main()
