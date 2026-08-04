import os
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    project_root = Path(__file__).resolve().parent
    
    # Input paths
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    kpis_csv = project_root / 'data' / 'credit_risk_kpis_master.csv'
    sentiment_csv = project_root / 'data' / 'qualitative_sentiment_scores.csv'
    
    if not labels_csv.exists() or not kpis_csv.exists() or not sentiment_csv.exists():
        logging.error("Missing required input CSVs. Check data/ directory.")
        return

    # 1. Load Labels
    logging.info("Loading final training labels...")
    labels_df = pd.read_csv(labels_csv)
    labels_df['CIK'] = labels_df['CIK'].astype(str).str.zfill(10)
    labels_df['Year'] = labels_df['Year'].astype(str)
    
    # Filter rated companies
    labels_df = labels_df[(labels_df['Rating'] != 'NR')]
    if 'SOURCE_INDICATOR' in labels_df.columns:
        labels_df = labels_df[labels_df['SOURCE_INDICATOR'].astype(str) != '0']

    # 2. Load KPIs
    logging.info("Loading quantitative KPIs...")
    kpis_df = pd.read_csv(kpis_csv)
    kpis_df['CIK_Identifier'] = kpis_df['CIK_Identifier'].astype(str).str.zfill(10)
    kpis_df['Fiscal_Year'] = kpis_df['Fiscal_Year'].astype(str)
    
    kpi_features = [
        'Debt-to-Equity', 'Retained Earnings / Total Assets', 'Current Ratio', 
        'Quick Ratio', 'Working Capital / Total Assets', 'ROCE', 
        'Net Profit Margin', 'Total Liabilities / Total Assets'
    ]
    kpis_df = kpis_df[['CIK_Identifier', 'Fiscal_Year'] + kpi_features]

    # 3. Load Qualitative Sentiment Scores
    logging.info("Loading qualitative sentiment scores...")
    try:
        sent_df = pd.read_csv(sentiment_csv)
        sent_df['Company_ID'] = sent_df['Company_ID'].astype(str).str.zfill(10)
        sent_df['Year'] = sent_df['Year'].astype(str)
        sent_features = [
            'Revenue_Sentiment', 'Operating Profit_Sentiment', 
            'Net/Gross Margins_Sentiment', 'Net Profit_Sentiment', 
            'Free Cash Flow_Sentiment'
        ]
    except Exception as e:
        logging.warning("Could not load qualitative_sentiment_scores.csv. Proceeding without FinBERT scores.")
        sent_df = pd.DataFrame()
        sent_features = []
    
    # 4. Merge all together
    logging.info("Merging datasets...")
    hybrid_df = pd.merge(labels_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    hybrid_df.drop(columns=['CIK_Identifier', 'Fiscal_Year'], inplace=True)
    
    if not sent_df.empty:
        hybrid_df = pd.merge(hybrid_df, sent_df, left_on=['CIK', 'Year'], right_on=['Company_ID', 'Year'], how='inner')
        hybrid_df.drop(columns=['Company_ID'], inplace=True)
    
    # 5. Format for OmegaSquared
    # OmegaSquared expects 'label' (for target) and temporal columns ('year', 'month' or 'year_month')
    hybrid_df.rename(columns={'Rating': 'label', 'Year': 'year'}, inplace=True)
    
    # Add a pseudo 'month' column (assuming 10-K is end of year) to satisfy temporal grouping logic
    hybrid_df['month'] = 12
    hybrid_df['year_month'] = hybrid_df['year'].astype(str) + "_12"
    
    final_cols = ['CIK', 'label', 'year', 'month', 'year_month'] + kpi_features
    
    # Only keep columns that exist in the dataframe
    final_cols = [c for c in final_cols if c in hybrid_df.columns]
    hybrid_df = hybrid_df[final_cols]
    
    # 6. Export to OmegaSquared directly
    hybrid_df = hybrid_df.drop_duplicates()
    
    omega_out_dir = project_root / 'OmegaSquared' / 'data' / 'CRED'
    omega_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = omega_out_dir / 'filteredpromax_kaggle_with_year_month_CRED.csv'
    
    logging.info(f"Exporting Omega2 formatted dataset with {len(hybrid_df)} records...")
    hybrid_df.to_csv(out_csv, index=False)
    
    logging.info(f"Success! Staged dataset saved to {out_csv}")

if __name__ == "__main__":
    main()
