import os
import requests
import time
import pandas as pd
from pathlib import Path

def get_edgar_data(cik_int, target_year, headers):
    # Pad CIK to 10 digits for the submissions API
    cik_str = str(cik_int).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    
    time.sleep(0.12)  # Strict SEC rate limit
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"Failed to fetch submissions for CIK {cik_str}")
        return None
        
    data = res.json()
    filings = data.get('filings', {}).get('recent', {})
    forms = filings.get('form', [])
    dates = filings.get('filingDate', [])
    acc_nums = filings.get('accessionNumber', [])
    
    for idx, form in enumerate(forms):
        if form == '10-K':
            year = dates[idx].split('-')[0]
            if year == str(target_year):
                return acc_nums[idx].replace('-', '')
                
    return None

def main():
    project_root = Path(__file__).resolve().parents[1]
    labels_csv = project_root / "data" / "cold_start_training_labels.csv"
    out_dir_rated = project_root / "data" / "KPI_tables"
    out_dir_nr = project_root / "data" / "KPI_tables_cold_start"
    
    out_dir_rated.mkdir(parents=True, exist_ok=True)
    out_dir_nr.mkdir(parents=True, exist_ok=True)
    
    # Custom headers required by SEC
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    
    print(f"Loading dataset from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    total = len(df)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for index, row in df.iterrows():
        cik = int(row['CIK'])
        year = row['Year']
        
        # Check if it's an NR company (SOURCE_INDICATOR == 0)
        is_nr = (str(row.get('SOURCE_INDICATOR', '1')) == '0')
        
        if is_nr:
            out_filepath = out_dir_nr / f"{cik}_{year}_10K.xlsx"
        else:
            out_filepath = out_dir_rated / f"{cik}_{year}_10K.xlsx"
        
        if out_filepath.exists():
            skipped += 1
            continue
            
        print(f"[{index+1}/{total}] Processing CIK {cik} for Year {year}...")
        
        # Get Accession Number
        acc_no = get_edgar_data(cik, year, headers)
        if not acc_no:
            print(f"  -> Could not find matching 10-K accession number.")
            failed += 1
            continue
            
        # Download Financial_Report.xlsx
        excel_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/Financial_Report.xlsx"
        time.sleep(0.12)
        
        doc_res = requests.get(excel_url, headers=headers)
        if doc_res.status_code == 200:
            with open(out_filepath, 'wb') as f:
                f.write(doc_res.content)
            downloaded += 1
            print(f"  -> Successfully downloaded Financial_Report.xlsx")
        else:
            print(f"  -> Failed to download Excel file (HTTP {doc_res.status_code})")
            failed += 1
            
    print("-" * 50)
    print("Download Summary:")
    print(f"Successfully Downloaded: {downloaded}")
    print(f"Skipped (Already Exists): {skipped}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
