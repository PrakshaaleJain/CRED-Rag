import os
import glob
import xml.etree.ElementTree as ET
import requests
import re
import difflib

# Attempt to load GPU-accelerated cuDF, fallback to pandas
try:
    import cudf as pd
    print("Successfully loaded cuDF for GPU acceleration.")
except ImportError:
    import pandas as pd
    print("cuDF not found, falling back to standard pandas.")

def normalize_name(name):
    if not name: return ""
    name = name.upper()
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    name = re.sub(r'\b(INC|CORP|CORPORATION|LLC|LP|LTD|PLC|COMPANY|CO)\b', '', name)
    return ' '.join(name.split())

def load_sec_mapping():
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    ticker_to_cik = {}
    name_to_cik = {}
    if res.status_code == 200:
        data = res.json()
        for k, v in data.items():
            cik_str = str(v['cik_str']).zfill(10)
            ticker_to_cik[v['ticker'].upper()] = cik_str
            norm_title = normalize_name(v['title'])
            if norm_title:
                name_to_cik[norm_title] = cik_str
    return ticker_to_cik, name_to_cik

def get_cik_for_xml(filepath, ticker_to_cik, name_to_cik, name_keys):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'r': 'http://xbrl.sec.gov/ratings/2015-03-31'}
    
    ticker_elem = root.find('.//r:ISI', ns)
    name_elem = root.find('.//r:ISSNAME', ns)
    ticker = ticker_elem.text if ticker_elem is not None else None
    name = name_elem.text if name_elem is not None else None
    
    cik = None
    if ticker and ticker not in ['ENT_01', 'NRSRO'] and ticker.upper() in ticker_to_cik:
        cik = ticker_to_cik[ticker.upper()]
    else:
        norm_name = normalize_name(name)
        if norm_name in name_to_cik:
            cik = name_to_cik[norm_name]
        elif norm_name:
            matches = difflib.get_close_matches(norm_name, name_keys, n=1, cutoff=0.95)
            if matches:
                cik = name_to_cik[matches[0]]
                
    return cik, ticker, name, root, ns

def extract_rating_history(root, ns):
    """
    Extracts a chronological dictionary of rating actions: { 'YYYY-MM-DD': 'RATING' }
    """
    history = {}
    for inrd in root.findall('.//r:INRD', ns):
        date_elem = inrd.find('r:RAD', ns)
        rating_elem = inrd.find('r:R', ns)
        
        if date_elem is not None and date_elem.text and rating_elem is not None and rating_elem.text:
            date_str = date_elem.text.strip()
            rating_str = rating_elem.text.strip()
            history[date_str] = rating_str
            
    # Return sorted chronologically
    return dict(sorted(history.items()))

def get_active_rating(history, target_year):
    """
    Finds the active rating as of Dec 31st of the target_year.
    """
    cutoff_date = f"{target_year}-12-31"
    active_rating = None
    
    for date, rating in history.items():
        if date <= cutoff_date:
            active_rating = rating
        else:
            break
            
    # If no rating existed before Dec 31st (e.g., first rating was in Jan of the following year),
    # fallback to the earliest rating available.
    if active_rating is None and history:
        active_rating = list(history.values())[0]
        
    return active_rating

def main():
    xml_dir = "data/egan_public"
    sec_dir = "data/egan_sec_filings_html"
    out_csv = "data/ground_truth_labels.csv"
    
    print("Loading SEC Mappings...")
    ticker_to_cik, name_to_cik = load_sec_mapping()
    name_keys = list(name_to_cik.keys())
    
    print("Scanning available downloaded SEC 10-K filings...")
    sec_files = glob.glob(os.path.join(sec_dir, "*.html"))
    cik_to_years = {}
    for f in sec_files:
        basename = os.path.basename(f)
        parts = basename.replace("_10-K.html", "").split('_')
        if len(parts) == 2:
            cik, year = parts[0], parts[1]
            if cik not in cik_to_years:
                cik_to_years[cik] = []
            cik_to_years[cik].append((year, basename))
            
    print(f"Found {len(sec_files)} SEC filings mapping to {len(cik_to_years)} distinct CIKs.")
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f"Parsing {len(xml_files)} Egan-Jones XML files for credit ratings...")
    
    dataset_rows = []
    
    for xml_file in xml_files:
        cik, ticker, name, root, ns = get_cik_for_xml(xml_file, ticker_to_cik, name_to_cik, name_keys)
        
        if cik and cik in cik_to_years:
            history = extract_rating_history(root, ns)
            
            for year, filename in cik_to_years[cik]:
                active_rating = get_active_rating(history, year)
                
                if active_rating:
                    dataset_rows.append({
                        "CIK": cik,
                        "Ticker": ticker if ticker else "UNKNOWN",
                        "Company_Name": name,
                        "Year": year,
                        "Rating": active_rating,
                        "10K_Filename": filename
                    })
                    
    print(f"\nExtraction complete. Found {len(dataset_rows)} valid data points.")
    
    # Create DataFrame (using cuDF if available, else pandas)
    df = pd.DataFrame(dataset_rows)
    
    # Save to CSV
    df.to_csv(out_csv, index=False)
    print(f"Saved ground truth dataset to {out_csv}\n")
    
    # ==========================================
    # VALIDATION LOGIC
    # ==========================================
    print("--- Running Validation ---")
    
    # 1. Check file existence
    missing_files = 0
    for row in dataset_rows:
        expected_path = os.path.join(sec_dir, row['10K_Filename'])
        if not os.path.exists(expected_path):
            missing_files += 1
            print(f"WARNING: File missing - {expected_path}")
            
    if missing_files == 0:
        print(f"SUCCESS: All {len(dataset_rows)} 10-K HTML files referenced in the CSV exist on disk.")
    else:
        print(f"FAILED: {missing_files} files are missing from {sec_dir}.")
        
    # 2. Print distribution of ratings
    print("\nRating Distribution in Dataset:")
    print(df['Rating'].value_counts())

if __name__ == '__main__':
    main()
