import os
import shutil
import csv
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Input files/directories
    labels_csv = project_root / 'data' / 'cold_start_training_labels.csv'
    kpi_rated_dir = project_root / 'data' / 'KPI_tables'
    kpi_nr_dir = project_root / 'data' / 'KPI_tables_cold_start'
    
    # Potential source directories for SEC filing HTMLs
    html_source_dirs = [
        project_root / 'data' / 'egan_sec_filings_html',
        project_root / 'data' / 'sec_filings_html',
        project_root / 'data' / 'egan_cold_start_html'
    ]
    
    # Output file/directory
    final_csv = project_root / 'data' / 'final_training_labels.csv'
    final_html_dir = project_root / 'data' / 'final_sec_filings'
    
    final_html_dir.mkdir(parents=True, exist_ok=True)
    
    with open(labels_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    headers = reader[0]
    data_rows = reader[1:]
    
    out_rows = [headers]
    
    kept_count = 0
    missing_kpi_count = 0
    html_moved_count = 0
    html_missing_count = 0
    
    for row in data_rows:
        cik = str(row[0]).zfill(10)
        year = str(row[3])
        source_indicator = str(row[-1])
        html_filename = row[5]
        
        # Check if KPI table exists
        kpi_filename = f"{int(cik)}_{year}_10K.xlsx"
        kpi_path_rated = kpi_rated_dir / kpi_filename
        kpi_path_nr = kpi_nr_dir / kpi_filename
        
        if kpi_path_rated.exists() or kpi_path_nr.exists():
            out_rows.append(row)
            kept_count += 1
            
            # Find and move HTML filing
            html_moved = False
            for s_dir in html_source_dirs:
                src_html = s_dir / html_filename
                if src_html.exists():
                    shutil.move(str(src_html), str(final_html_dir / html_filename))
                    html_moved = True
                    html_moved_count += 1
                    break
            
            if not html_moved:
                # Also check if it's already in the final_html_dir
                if (final_html_dir / html_filename).exists():
                    html_moved_count += 1
                else:
                    html_missing_count += 1
                    print(f"Warning: Could not find HTML file {html_filename} to move.")
        else:
            missing_kpi_count += 1
            
    # Save the final dataset
    with open(final_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
        
    print(f"Processing complete.")
    print(f"Kept {kept_count} rows with downloaded KPIs.")
    print(f"Removed {missing_kpi_count} rows that failed KPI download.")
    print(f"Moved {html_moved_count} SEC HTML filings to {final_html_dir}.")
    if html_missing_count > 0:
        print(f"Failed to find/move {html_missing_count} SEC HTML filings.")
    print(f"Saved final dataset to {final_csv}")

if __name__ == '__main__':
    main()
