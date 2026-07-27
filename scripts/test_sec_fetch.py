import os
import glob
import xml.etree.ElementTree as ET
import requests
import time

def main():
    print("Fetching SEC ticker-to-CIK mapping...")
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    ticker_to_cik = {}
    if res.status_code == 200:
        data = res.json()
        for k, v in data.items():
            ticker_to_cik[v['ticker'].upper()] = str(v['cik_str']).zfill(10)
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
    
    for i, f in enumerate(xml_files):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i}/{len(xml_files)} files...")

        ticker, name, years = parse_egan_xml(f)
        if not ticker or ticker in ['ENT_01', 'NRSRO']: 
            unmatched_count += 1
            continue
            
        if ticker.upper() in ticker_to_cik:
            cik = ticker_to_cik[ticker.upper()]
            
            # Fetch submissions
            sub_res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
            if sub_res.status_code == 200:
                sub_data = sub_res.json()
                filings = sub_data.get('filings', {}).get('recent', {})
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                
                # Extract 10-K filing dates
                found_10ks = [dates[idx] for idx, form in enumerate(forms) if form == '10-K']
                
                # Get the latest 2 years of 10-K filings
                latest_2_10ks = found_10ks[:2]
                
                filing_years = set(d.split('-')[0] for d in latest_2_10ks)
                rating_years = set(years)
                overlap = rating_years.intersection(filing_years)
                
                if overlap:
                    matched_count += 1
                    # print(f"Matched: {name} (Ticker: {ticker.upper()}) | Latest 10-Ks: {latest_2_10ks} | Overlap: {sorted(list(overlap))}")
                else:
                    unmatched_count += 1
            else:
                unmatched_count += 1
            time.sleep(0.12) # Respect SEC rate limits (max 10 req/sec)
        else:
            unmatched_count += 1
            
    print("-" * 60)
    print(f"Execution Completed.")
    print(f"Total Matched (with recent 10-K overlap): {matched_count}")
    print(f"Total Unmatched/Skipped: {unmatched_count}")

if __name__ == '__main__':
    main()
