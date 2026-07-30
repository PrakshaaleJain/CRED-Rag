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
    gt_map = {}
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        # The header might have an accidental '1' from a previous script. We only want the first 6.
        headers = reader[0][:6] 
        for row in reader[1:]:
            if len(row) >= 5:
                cik = str(row[0]).zfill(10)
                year = str(row[3])
                gt_map[(cik, year)] = row[:6] # Keep only the first 6 original columns

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
            filename = f"{cik}_{year}_10-K.html"
            
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
                # File is missing from ground_truth_labels entirely.
                # We add it as an NR (Not Rated) entry to ensure we hit the 2462 total.
                row = [cik, "UNKNOWN", "UNKNOWN", year, "NR", filename]
                new_row = row + ['0'] # SOURCE_INDICATOR = 0
                out_rows.append(new_row)

    with open(out_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
        
    print(f"\nCreated dataset with {len(out_rows)-1} records at {out_csv}")
    print(f"Rated (SOURCE_INDICATOR=1): {matched_rated}")
    print(f"NR (SOURCE_INDICATOR=0): {matched_nr}")
    print(f"Missing from ground truth (assigned as NR): {missing}")
    print(f"Total Rows Saved: {len(out_rows)-1}")

if __name__ == '__main__':
    main()
