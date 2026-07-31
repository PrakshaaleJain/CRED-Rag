from pathlib import Path
import os
import csv

project_root = Path(__file__).resolve().parent
labels_csv = project_root / 'data' / 'final_training_labels.csv'
features_dir = project_root / 'data' / 'qualitative_features'
trees_dir = project_root / 'data' / 'final_trees_directory'

with open(labels_csv, mode='r') as f:
    reader = list(csv.reader(f))
data_rows = reader[1:]

missing = []
for row in data_rows:
    cik = str(row[0]).zfill(10)
    year = str(row[3])
    rating = str(row[4])
    source_indicator = str(row[6]) if len(row) > 6 else '1'
    
    if rating == 'NR' or source_indicator == '0': continue
    
    out_file = features_dir / f"{cik}_{year}_features.json"
    if not out_file.exists():
        # Check if tree exists
        tree_folder = trees_dir / f"{cik}_{year}_10-K_extracted"
        if tree_folder.exists():
            missing.append(f"{cik}_{year}")
            
print(f"Found {len(missing)} missing companies with existing trees:")
for m in missing:
    print(m)
