import csv
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    input_csv = project_root / 'data' / 'final_training_labels.csv'
    output_csv = project_root / 'data' / 'unrated_inference_set.csv'
    
    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        return
        
    print(f"Reading {input_csv}...")
    with open(input_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    header = reader[0]
    data_rows = reader[1:]
    
    inference_rows = []
    
    for row in data_rows:
        rating = str(row[4])
        source_indicator = str(row[6]) if len(row) > 6 else '1'
        
        # We exclusively want the companies that were SKIPPED by the training pipeline
        if rating == 'NR' or source_indicator == '0':
            inference_rows.append(row)
            
    print(f"Found {len(inference_rows)} unrated companies for the inference set.")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(inference_rows)
        
    print(f"Successfully saved to {output_csv}")

if __name__ == '__main__':
    main()
