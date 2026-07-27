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
    
    count = 0
    # Let's test until we successfully process 3 public companies
    for f in xml_files:
        ticker, name, years = parse_egan_xml(f)
        if not ticker or ticker == 'ENT_01' or ticker == 'NRSRO': 
            continue
            
        if ticker.upper() in ticker_to_cik:
            count += 1
            print(f"File: {os.path.basename(f)}")
            print(f"Company: {name}, Ticker: {ticker.upper()}, Rating Years: {years}")
            
            cik = ticker_to_cik[ticker.upper()]
            print(f"Matched to SEC CIK: {cik}")
            
            # Fetch submissions
            sub_res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
            if sub_res.status_code == 200:
                sub_data = sub_res.json()
                filings = sub_data.get('filings', {}).get('recent', {})
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                
                # check for 10-K in the years
                found_10ks = []
                for i, form in enumerate(forms):
                    if form == '10-K':
                        found_10ks.append(dates[i])
                        
                print(f"Found 10-K filings on dates: {found_10ks[:5]} ... (showing first 5)")
                
                # Verify if we have 10-Ks matching the rating years
                filing_years = set(d.split('-')[0] for d in found_10ks)
                rating_years = set(years)
                overlap = rating_years.intersection(filing_years)
                print(f"Overlap between Rating Years and 10-K Filing Years: {sorted(list(overlap))}")
            else:
                print(f"Failed to fetch SEC submissions for CIK {cik}")
            print("-" * 60)
            time.sleep(0.5) # rate limit
            
        if count >= 3:
            break

if __name__ == '__main__':
    main()
