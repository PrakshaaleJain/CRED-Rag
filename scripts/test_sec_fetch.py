import os
import glob
import xml.etree.ElementTree as ET
import requests
import time
import re
import difflib

def normalize_name(name):
    if not name: return ""
    name = name.upper()
    # Remove special characters
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    # Remove common corporate suffixes
    name = re.sub(r'\b(INC|CORP|CORPORATION|LLC|LP|LTD|PLC|COMPANY|CO)\b', '', name)
    return ' '.join(name.split())

def main():
    print("Fetching SEC ticker-to-CIK mapping...")
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    ticker_to_cik = {}
    name_to_cik = {}
    if res.status_code == 200:
        data = res.json()
        for k, v in data.items():
            cik_str = str(v['cik_str']).zfill(10)
            ticker_to_cik[v['ticker'].upper()] = cik_str
            
            # Create a normalized name index for fuzzy matching fallback
            norm_title = normalize_name(v['title'])
            if norm_title:
                name_to_cik[norm_title] = cik_str
    else:
        print("Failed to fetch SEC tickers.")
        return

    def parse_egan_xml(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {'r': 'http://xbrl.sec.gov/ratings/2015-03-31'}
        
        # Extract ticker (ISI) and name
        ticker_elem = root.find('.//r:ISI', ns)
        name_elem = root.find('.//r:ISSNAME', ns)
        
        ticker = ticker_elem.text if ticker_elem is not None else None
        name = name_elem.text if name_elem is not None else None
        
        # Extract years from rating actions
        years = set()
        for inrd in root.findall('.//r:INRD', ns):
            rad_elem = inrd.find('r:RAD', ns)
            if rad_elem is not None and rad_elem.text:
                years.add(rad_elem.text.split('-')[0])
                
        return ticker, name, sorted(list(years))

    xml_files = glob.glob('data/Egan/*.xml')
    print(f"Found {len(xml_files)} XML files.")
    
    matched_count = 0
    unmatched_count = 0
    
    # Extract keys once to speed up fuzzy matching
    name_keys = list(name_to_cik.keys())
    
    for i, f in enumerate(xml_files):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(xml_files)} files...")

        ticker, name, years = parse_egan_xml(f)
        
        # Step 1: Match by Ticker first
        cik = None
        if ticker and ticker not in ['ENT_01', 'NRSRO'] and ticker.upper() in ticker_to_cik:
            cik = ticker_to_cik[ticker.upper()]
        else:
            # Step 2: Fallback to normalized Issuer Name
            norm_name = normalize_name(name)
            if norm_name in name_to_cik:
                cik = name_to_cik[norm_name]
            elif norm_name:
                # Step 3: Fuzzy name matching
                matches = difflib.get_close_matches(norm_name, name_keys, n=1, cutoff=0.85)
                if matches:
                    cik = name_to_cik[matches[0]]
                
        if cik:
            # Fetch submissions
            sub_res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
            if sub_res.status_code == 200:
                sub_data = sub_res.json()
                filings = sub_data.get('filings', {}).get('recent', {})
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                
                # Extract ALL 10-K filing dates (strictly 10-K as requested)
                found_10ks = [dates[idx] for idx, form in enumerate(forms) if form == '10-K']
                
                filing_years = set(d.split('-')[0] for d in found_10ks)
                rating_years = set(years)
                overlap = rating_years.intersection(filing_years)
                
                if overlap:
                    matched_count += 1
                else:
                    unmatched_count += 1
            else:
                unmatched_count += 1
            time.sleep(0.12) # Respect SEC rate limits (max 10 req/sec)
        else:
            unmatched_count += 1
            
    print("-" * 60)
    print(f"Execution Completed.")
    print(f"Total Matched (with any 10-K overlap): {matched_count}")
    print(f"Total Unmatched/Skipped: {unmatched_count}")

if __name__ == '__main__':
    main()
