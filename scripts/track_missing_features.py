import os
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    features_dir = project_root / 'data' / 'qualitative_features'
    trees_dir = project_root / 'data' / 'final_trees_directory'
    
    if not trees_dir.exists():
        print(f"Error: {trees_dir} does not exist on this machine.")
        return
        
    print(f"Scanning for missing features...")
    
    failed_companies = []
    
    for tree_folder in trees_dir.iterdir():
        if tree_folder.is_dir() and tree_folder.name.endswith('_10-K_extracted'):
            # Extract CIK and Year from folder name
            parts = tree_folder.name.replace('_10-K_extracted', '').split('_')
            if len(parts) == 2:
                cik, year = parts
                out_file = features_dir / f"{cik}_{year}_features.json"
                
                if not out_file.exists():
                    failed_companies.append(f"{cik}_{year}")
                    
    print("\n--- Summary ---")
    print(f"Total missing extractions (trees exist but JSON failed): {len(failed_companies)}")
    
    if failed_companies:
        print("\nList of failed companies (JSON Error):")
        for company in failed_companies:
            print(f"- {company}")
            
if __name__ == '__main__':
    main()
