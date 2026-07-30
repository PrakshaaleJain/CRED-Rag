import os
import shutil
import csv
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Input files/directories
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    source_trees_dir = project_root / 'data' / 'egan_sec_filings_trees'
    
    # Output directory
    final_trees_dir = project_root / 'data' / 'final_trees_directory'
    
    # Create the output directory if it doesn't exist
    final_trees_dir.mkdir(parents=True, exist_ok=True)
    
    if not labels_csv.exists():
        print(f"Error: {labels_csv} not found.")
        return
        
    print(f"Reading {labels_csv}...")
    with open(labels_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    # Assume headers are present
    data_rows = reader[1:]
    
    moved_count = 0
    missing_count = 0
    skipped_nr_count = 0
    
    for row in data_rows:
        cik = str(row[0]).zfill(10)
        year = str(row[3])
        rating = str(row[4])
        source_indicator = str(row[6]) if len(row) > 6 else '1'
        
        # We only want to move trees for rated companies
        if rating == 'NR' or source_indicator == '0':
            skipped_nr_count += 1
            continue
            
        # The tree directories are named like: {cik}_{year}_10-K_extracted
        tree_folder_name = f"{int(cik)}_{year}_10-K_extracted"
        src_tree_path = source_trees_dir / tree_folder_name
        dest_tree_path = final_trees_dir / tree_folder_name
        
        if src_tree_path.exists() and src_tree_path.is_dir():
            shutil.move(str(src_tree_path), str(dest_tree_path))
            moved_count += 1
        else:
            # Maybe the folder doesn't have the "_extracted" suffix, or it's named slightly differently
            # Let's try an alternative naming scheme just in case
            alt_tree_folder_name = f"{cik}_{year}_10-K_extracted"
            alt_src_tree_path = source_trees_dir / alt_tree_folder_name
            
            if alt_src_tree_path.exists() and alt_src_tree_path.is_dir():
                shutil.move(str(alt_src_tree_path), str(final_trees_dir / alt_tree_folder_name))
                moved_count += 1
            elif dest_tree_path.exists():
                # Already moved in a previous run
                moved_count += 1
            else:
                missing_count += 1
                print(f"Warning: Could not find tree directory for CIK {cik} Year {year}")
                
    print("\n--- Summary ---")
    print(f"Moved {moved_count} tree folders to {final_trees_dir}")
    print(f"Skipped {skipped_nr_count} NR companies")
    if missing_count > 0:
        print(f"Missing {missing_count} tree folders from {source_trees_dir}")

if __name__ == '__main__':
    main()
