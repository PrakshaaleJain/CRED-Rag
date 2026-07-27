import os
import glob
import xml.etree.ElementTree as ET
import requests
import time
import re
import difflib
import shutil

def normalize_name(name):
    if not name: return ""
    name = name.upper()
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    name = re.sub(r'\b(INC|CORP|CORPORATION|LLC|LP|LTD|PLC|COMPANY|CO)\b', '', name)
    return ' '.join(name.split())

def main():
    # SEC strictly enforces a 10 requests/second limit and blocks requests without a custom User-Agent
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    
    # Setup directories
    xml_out_dir = "data/egan_public"
    sec_out_dir = "data/egan_sec_filings_html"
    os.makedirs(xml_out_dir, exist_ok=True)
    os.makedirs(sec_out_dir, exist_ok=True)
    
    print("Fetching SEC ticker-to-CIK mapping...")
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
    else:
        print("Failed to fetch SEC tickers.")
        return

    def parse_egan_xml(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {'r': 'http://xbrl.sec.gov/ratings/2015-03-31'}
        ticker_elem = root.find('.//r:ISI', ns)
        name_elem = root.find('.//r:ISSNAME', ns)
        ticker = ticker_elem.text if ticker_elem is not None else None
        name = name_elem.text if name_elem is not None else None
        
        years = set()
        for inrd in root.findall('.//r:INRD', ns):
            rad_elem = inrd.find('r:RAD', ns)
            if rad_elem is not None and rad_elem.text:
                years.add(rad_elem.text.split('-')[0])
                
        return ticker, name, sorted(list(years))

    xml_files = glob.glob('data/Egan/*.xml')
    print(f"Found {len(xml_files)} XML files.")
    
    name_keys = list(name_to_cik.keys())
    matched_count = 0
    downloaded_filings = 0
    
    for i, f in enumerate(xml_files):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(xml_files)} files. Matched: {matched_count}, Filings downloaded: {downloaded_filings}")

        ticker, name, years = parse_egan_xml(f)
        
        cik = None
        if ticker and ticker not in ['ENT_01', 'NRSRO'] and ticker.upper() in ticker_to_cik:
            cik = ticker_to_cik[ticker.upper()]
        else:
            norm_name = normalize_name(name)
            if norm_name in name_to_cik:
                cik = name_to_cik[norm_name]
            elif norm_name:
                matches = difflib.get_close_matches(norm_name, name_keys, n=1, cutoff=0.85)
                if matches:
                    cik = name_to_cik[matches[0]]
                
        if cik:
            # Fetch submissions metadata
            sub_res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
            if sub_res.status_code == 200:
                sub_data = sub_res.json()
                filings = sub_data.get('filings', {}).get('recent', {})
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                acc_nums = filings.get('accessionNumber', [])
                primary_docs = filings.get('primaryDocument', [])
                
                # Extract 10-K filing metadata
                found_10ks = []
                for idx, form in enumerate(forms):
                    if form == '10-K':
                        found_10ks.append({
                            'year': dates[idx].split('-')[0],
                            'acc': acc_nums[idx],
                            'doc': primary_docs[idx]
                        })
                
                filing_years = set(f['year'] for f in found_10ks)
                rating_years = set(years)
                overlap = rating_years.intersection(filing_years)
                
                if overlap:
                    matched_count += 1
                    
                    # 1. Copy the matched XML file to egan_public
                    xml_basename = os.path.basename(f)
                    shutil.copy2(f, os.path.join(xml_out_dir, xml_basename))
                    
                    # 2. Download overlapping SEC 10-Ks
                    # Group by year to only download one 10-K per overlapping year
                    for year in overlap:
                        # Find the specific filing data for this year
                        filing = next(item for item in found_10ks if item['year'] == year)
                        
                        acc_no_dashes = filing['acc'].replace('-', '')
                        doc = filing['doc']
                        cik_int = int(cik) # EDGAR URLs require CIK without leading zeros
                        
                        # Construct the EDGAR Archive URL
                        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{doc}"
                        
                        # Save it as CIK_YEAR_10-K.html
                        out_filepath = os.path.join(sec_out_dir, f"{cik}_{year}_10-K.html")
                        
                        # Skip if already downloaded (allows pausing/restarting the script)
                        if not os.path.exists(out_filepath):
                            try:
                                time.sleep(0.12) # Strict SEC rate limit
                                doc_res = requests.get(url, headers=headers)
                                if doc_res.status_code == 200:
                                    with open(out_filepath, 'wb') as out_f:
                                        out_f.write(doc_res.content)
                                    downloaded_filings += 1
                                else:
                                    print(f"Failed to download {url} - Status: {doc_res.status_code}")
                            except Exception as e:
                                print(f"Error downloading {url}: {e}")
            
            # Rate limit for submission JSON requests
            time.sleep(0.12) 
            
    print("-" * 60)
    print(f"Execution Completed.")
    print(f"Total Matched Companies (XMLs copied): {matched_count}")
    print(f"Total SEC Filings Downloaded: {downloaded_filings}")

if __name__ == '__main__':
    main()
