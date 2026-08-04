import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    project_root = Path(__file__).resolve().parent
    
    omega2_csv = project_root / 'scratch' / 'omega2_data' / 'corporateCreditRatingWithFinancialRatios.csv'
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    kpis_csv = project_root / 'data' / 'credit_risk_kpis_master.csv'
    
    if not omega2_csv.exists():
        logging.error("Omega2 dataset not found. Please unzip it first.")
        return
        
    logging.info("Loading datasets...")
    omega_df = pd.read_csv(omega2_csv)
    labels_df = pd.read_csv(labels_csv)
    kpis_df = pd.read_csv(kpis_csv)
    
    # Preprocess Omega2 CIK and Year
    omega_df['CIK'] = omega_df['CIK'].astype(str).str.zfill(10)
    omega_df['Year'] = pd.to_datetime(omega_df['Rating Date']).dt.year.astype(str)
    
    # Preprocess User data
    labels_df['CIK'] = labels_df['CIK'].astype(str).str.zfill(10)
    labels_df['Year'] = labels_df['Year'].astype(str)
    kpis_df['CIK'] = kpis_df['CIK_Identifier'].astype(str).str.zfill(10)
    kpis_df['Year'] = kpis_df['Fiscal_Year'].astype(str)
    
    # Merge user data first to get user's Rating + KPIs in one place
    user_df = pd.merge(labels_df[['CIK', 'Year', 'Rating']], kpis_df, on=['CIK', 'Year'], how='inner')
    
    # Find intersecting rows
    logging.info("Finding intersection on CIK and Year...")
    intersect_df = pd.merge(omega_df, user_df, on=['CIK', 'Year'], how='inner', suffixes=('_omega', '_user'))
    logging.info(f"Found {len(intersect_df)} overlapping records!")
    
    # Keep only intersecting companies (or we could keep all, but standard is intersection)
    final_df = intersect_df.copy()
    
    # 2. Label Resolution
    # Replace label if they don't match
    mismatched_labels = final_df['Rating_omega'] != final_df['Rating_user']
    logging.info(f"Overwriting {mismatched_labels.sum()} mismatched labels with user's 22-notch labels.")
    final_df['label'] = final_df['Rating_user']
    
    # We also need a 'year' and 'month' column for the Omega2 temporal framework
    final_df['year'] = final_df['Year']
    final_df['month'] = 12
    final_df['year_month'] = final_df['year'].astype(str) + "_12"
    
    # 3. Feature Mapping
    # The 24 numerical columns (we have 16 visible from the head, we will identify all numeric)
    # Omega2 metadata columns to keep
    meta_cols = ['Rating Agency', 'Corporation', 'Rating Date', 'CIK', 'Binary Rating', 'SIC Code', 'Sector', 'Ticker']
    
    # Identify the original Omega2 numerical ratio columns
    omega_ratio_cols = [c for c in omega_df.columns if c not in meta_cols and c not in ['Rating', 'Year']]
    
    mapping = {
        'Current Ratio': 'Current Ratio_user',
        'Debt/Equity Ratio': 'Debt-to-Equity',
        'Net Profit Margin': 'Net Profit Margin_user',
        'ROI - Return On Investment': 'ROCE',
        'Long-term Debt / Capital': 'Total Liabilities / Total Assets'
    }
    
    mapped_count = 0
    nulled_count = 0
    
    for col in omega_ratio_cols:
        if col in mapping and mapping[col] in final_df.columns:
            # Overwrite with user's value
            final_df[col] = final_df[mapping[col]]
            mapped_count += 1
        else:
            # Overwrite with NULL
            final_df[col] = np.nan
            nulled_count += 1
            
    logging.info(f"Mapped {mapped_count} columns to User KPIs. Nulled {nulled_count} columns.")
    
    # Keep only the columns expected by Omega2 (meta + label + year + month + year_month + 24 ratios)
    out_cols = meta_cols + ['label', 'year', 'month', 'year_month'] + omega_ratio_cols
    # Ensure all out_cols exist
    final_out_df = final_df[[c for c in out_cols if c in final_df.columns]]
    final_out_df = final_out_df.drop_duplicates()
    
    # Save the file
    omega_out_dir = project_root / 'OmegaSquared' / 'data' / 'CRED'
    omega_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = omega_out_dir / 'filteredpromax_kaggle_with_year_month_CRED.csv'
    
    final_out_df.to_csv(out_csv, index=False)
    logging.info(f"Saved highly-mapped intersection dataset to {out_csv}")

if __name__ == "__main__":
    main()
