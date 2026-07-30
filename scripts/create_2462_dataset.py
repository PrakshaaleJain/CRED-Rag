import csv
import os
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / 'data' / 'ground_truth_labels.csv'
    extracted_text_dir = project_root / 'data' / 'egan_sec_filings_extracted_text'
    out_csv = project_root / 'data' / 'cold_start_training_labels.csv'
    
    if not extracted_text_dir.exists():
        print(f"Error: Could not find extracted text directory at {extracted_text_dir}")
        print("Please run this script on the server where the 2462 extracted JSON files are located.")
        return

    # Read all ground truth labels into a dictionary mapped by (CIK, Year)
    # ground_truth_labels.csv has headers: CIK,Ticker,Company_Name,Year,Rating,10K_Filename
    gt_map = {}
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        headers = reader[0]
        for row in reader[1:]:
            if len(row) >= 5:
                cik = str(row[0]).zfill(10)
                year = str(row[3])
                gt_map[(cik, year)] = row

    # Iterate over the extracted JSON files
    extracted_files = list(extracted_text_dir.glob("*_extracted.json"))
    print(f"Found {len(extracted_files)} extracted files in {extracted_text_dir}")
    
    out_rows = []
    # Add new header
    out_rows.append(headers + ["SOURCE_INDICATOR"])
    
    matched_rated = 0
    matched_nr = 0
    missing = 0
    
    for file_path in extracted_files:
        # Filename format: {cik}_{year}_10-K_extracted.json
        basename = file_path.name
        parts = basename.split('_')
        if len(parts) >= 2:
            cik = str(parts[0]).zfill(10)
            year = str(parts[1])
            
            if (cik, year) in gt_map:
                row = gt_map[(cik, year)]
                rating = row[4]
                
                if rating == 'NR':
                    source_indicator = '0'
                    matched_nr += 1
                else:
                    source_indicator = '1'
                    matched_rated += 1
                    
                new_row = row + [source_indicator]
                out_rows.append(new_row)
            else:
                missing += 1
                print(f"Warning: Extracted file {basename} not found in ground truth labels.")

    with open(out_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
        
    print(f"\nCreated dataset with {len(out_rows)-1} records at {out_csv}")
    print(f"Rated (SOURCE_INDICATOR=1): {matched_rated}")
    print(f"NR (SOURCE_INDICATOR=0): {matched_nr}")
    print(f"Missing from ground truth: {missing}")
    if (matched_rated + matched_nr) != 2462:
        print(f"Note: Total found ({(matched_rated + matched_nr)}) does not match expected 2462.")

if __name__ == '__main__':
    main()
