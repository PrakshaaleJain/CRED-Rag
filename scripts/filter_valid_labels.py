import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # Paths
    extracted_dir = project_root / "data" / "egan_sec_filings_extracted_text"
    final_labels_csv = project_root / "data" / "final_training_labels.csv"
    output_csv = project_root / "data" / "valid_training_labels.csv"
    
    print(f"Loading existing labels from {final_labels_csv}")
    df_labels = pd.read_csv(final_labels_csv)
    
    # Get all the valid JSON filenames that survived the purge
    json_files = list(extracted_dir.glob("*_extracted.json"))
    print(f"Found {len(json_files)} valid extracted JSON files.")
    
    # The JSON files are named like '0000078239_2020_10-K_extracted.json'
    # The CSV uses '10K_Filename' like '0000078239_2020_10-K.html'
    # Let's map the JSON filename to the HTML filename format so we can join them
    valid_html_filenames = [f.name.replace("_extracted.json", ".html") for f in json_files]
    
    # Filter the DataFrame to only include rows where the 10K_Filename is in our valid list
    df_valid = df_labels[df_labels['10K_Filename'].isin(valid_html_filenames)]
    
    print(f"Filtered dataset from {len(df_labels)} down to {len(df_valid)} perfectly matched records.")
    
    # Save the strictly valid training set
    df_valid.to_csv(output_csv, index=False)
    print(f"Saved cleanly mapped dataset to {output_csv}")

if __name__ == "__main__":
    main()
