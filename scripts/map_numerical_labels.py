import pandas as pd
import os

def main():
    input_csv = "data/ground_truth_labels.csv"
    output_csv = "data/final_training_labels.csv"
    
    # 21-point ordinal scale for credit ratings (1 = Best/Safest, 22 = Default/Worst)
    rating_map = {
        'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
        'A+': 5, 'A': 6, 'A-': 7,
        'BBB+': 8, 'BBB': 9, 'BBB-': 10,
        'BB+': 11, 'BB': 12, 'BB-': 13,
        'B+': 14, 'B': 15, 'B-': 16,
        'CCC+': 17, 'CCC': 18, 'CCC-': 19,
        'CC': 20, 'C': 21, 'D': 22
    }
    
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}. Please ensure it exists on this machine.")
        return
        
    df = pd.read_csv(input_csv)
    initial_len = len(df)
    
    # 1. Drop duplicate rows caused by subsidiary XMLs mapping to the same parent CIK
    df = df.drop_duplicates(subset=['CIK', 'Year'])
    print(f"Dropped {initial_len - len(df)} duplicate entity mappings.")
    
    # 2. Drop 'NR' (Not Rated) rows
    nr_count = len(df[df['Rating'] == 'NR'])
    df = df[df['Rating'] != 'NR']
    print(f"Dropped {nr_count} 'NR' (Not Rated) rows.")
    
    # 3. Create the numerical Risk_Score mapping
    df['Risk_Score'] = df['Rating'].map(rating_map)
    
    # Drop any stray ratings that weren't in our dictionary
    missing = df['Risk_Score'].isna().sum()
    if missing > 0:
        print(f"Dropped {missing} rows with unrecognized rating formats.")
        df = df.dropna(subset=['Risk_Score'])
        
    # Convert score to integer
    df['Risk_Score'] = df['Risk_Score'].astype(int)
    
    # Reorder columns so the mapping is explicitly clear
    df = df[['CIK', 'Ticker', 'Company_Name', 'Year', '10K_Filename', 'Rating', 'Risk_Score']]
    
    df.to_csv(output_csv, index=False)
    print(f"\nFinal dataset contains {len(df)} clean, unique, numerically-mapped 10-K pairs.")
    print(f"Saved to: {output_csv}")

if __name__ == '__main__':
    main()
