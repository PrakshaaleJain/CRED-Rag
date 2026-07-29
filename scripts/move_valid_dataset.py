import pandas as pd
import shutil
from pathlib import Path
import os

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Paths
    labels_csv = project_root / "data" / "valid_training_labels.csv"
    source_dir = project_root / "data" / "egan_sec_filings_extracted_text"
    dest_dir = project_root / "data" / "final_training_dataset"
    
    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading valid labels from {labels_csv}")
    df_labels = pd.read_csv(labels_csv)
    
    moved_count = 0
    missing_count = 0
    
    print(f"Copying files to {dest_dir} ...")
    
    for _, row in df_labels.iterrows():
        # Map back to the JSON filename
        # 10K_Filename looks like '0000078239_2016_10-K.html'
        json_filename = row['10K_Filename'].replace(".html", "_extracted.json")
        
        source_path = source_dir / json_filename
        dest_path = dest_dir / json_filename
        
        if source_path.exists():
            # We use copy2 to preserve metadata, just to be safe
            shutil.copy2(source_path, dest_path)
            moved_count += 1
        else:
            print(f"WARNING: File not found: {source_path}")
            missing_count += 1
            
    print(f"\nOperation complete!")
    print(f"Successfully copied {moved_count} files.")
    if missing_count > 0:
        print(f"Failed to find {missing_count} files.")

if __name__ == "__main__":
    main()
