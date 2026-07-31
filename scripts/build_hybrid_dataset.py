import os
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Input paths
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    kpis_csv = project_root / 'data' / 'credit_risk_kpis_master.csv'
    sentiment_csv = project_root / 'data' / 'qualitative_sentiment_scores.csv'
    
    # Output path
    hybrid_out_csv = project_root / 'data' / 'hybrid_training_features.csv'
    
    if not sentiment_csv.exists():
        logging.error(f"{sentiment_csv} not found! Please run the sentiment analysis script first.")
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
    # Drop irrelevant columns from kpis to avoid duplicates
    kpis_df = kpis_df[['CIK_Identifier', 'Fiscal_Year'] + kpi_features]

    # 3. Load Qualitative Sentiment Scores
    logging.info("Loading qualitative sentiment scores...")
    sent_df = pd.read_csv(sentiment_csv)
    sent_df['Company_ID'] = sent_df['Company_ID'].astype(str).str.zfill(10)
    sent_df['Year'] = sent_df['Year'].astype(str)
    
    sent_features = [
        'Revenue_Sentiment', 'Operating Profit_Sentiment', 
        'Net/Gross Margins_Sentiment', 'Net Profit_Sentiment', 
        'Free Cash Flow_Sentiment'
    ]
    
    # 4. Merge all together
    logging.info("Merging datasets...")
    
    # Merge Labels + KPIs
    hybrid_df = pd.merge(labels_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    hybrid_df.drop(columns=['CIK_Identifier', 'Fiscal_Year'], inplace=True)
    
    # Merge (Labels+KPIs) + Sentiment
    hybrid_df = pd.merge(hybrid_df, sent_df, left_on=['CIK', 'Year'], right_on=['Company_ID', 'Year'], how='inner')
    hybrid_df.drop(columns=['Company_ID'], inplace=True)
    
    # Reorder columns nicely
    metadata_cols = ['CIK', 'Year', 'Rating']
    if 'SOURCE_INDICATOR' in hybrid_df.columns:
        metadata_cols.append('SOURCE_INDICATOR')
        
    final_cols = metadata_cols + kpi_features + sent_features
    
    # Keep only the columns we explicitly asked for, plus any extra metadata safely
    hybrid_df = hybrid_df[final_cols]
    
    # 5. Export
    logging.info(f"Exporting Hybrid Dataset with {len(hybrid_df)} records and {len(kpi_features) + len(sent_features)} features...")
    hybrid_df.to_csv(hybrid_out_csv, index=False)
    
    logging.info(f"Success! Hybrid dataset saved to {hybrid_out_csv}")
    print("\nSample Output (First 3 Rows):")
    print(hybrid_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
