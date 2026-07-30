import csv
import os
import shutil
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / 'data' / 'ground_truth_labels.csv'
    dest_dir = project_root / 'data' / 'egan_cold_start_html'
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory to find files
    source_dir = project_root / 'data' / 'egan_sec_filings_html'
    
    moved_count = 0
    not_found_count = 0
    
    # Read the original CSV
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    updated_rows = []
    
    for row in reader:
        # Expected row structure: CIK, Ticker, Company, Year, Rating, Filename
        if len(row) >= 5:
            rating = row[4]
            filename = row[5] if len(row) > 5 else None
            
            if rating == 'NR':
                source_indicator = '0'
                
                # Attempt to find and move the file
                if filename:
                    src_file = source_dir / filename
                    if src_file.exists():
                        shutil.move(str(src_file), str(dest_dir / filename))
                        moved_count += 1
                    else:
                        not_found_count += 1
            else:
                source_indicator = '1'
                
            # Append SOURCE INDICATOR
            new_row = row[:6] + [source_indicator]
            updated_rows.append(new_row)
        else:
            updated_rows.append(row)
            
    # Save the updated CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)
        
    print("Done processing CSV.")
    print(f"Moved {moved_count} NR files to {dest_dir}")
    print(f"Could not find {not_found_count} NR files in {source_dir}.")
    print("Added SOURCE INDICATOR column to ground_truth_labels.csv")

if __name__ == '__main__':
    main()
