import pandas as pd
from pathlib import Path
import json

def explore_excel(file_path):
    print(f"\n{'='*50}\nExploring {file_path}\n{'='*50}")
    
    # Get all sheet names
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    print(f"Sheet Names:\n{json.dumps(sheet_names, indent=2)}\n")
    
    # Identify potential Balance Sheet and Income Statement sheets
    bs_sheet = None
    is_sheet = None
    
    for sheet in sheet_names:
        sheet_lower = sheet.lower()
        if "balance" in sheet_lower and "sheet" in sheet_lower:
            bs_sheet = sheet
        if ("statement" in sheet_lower and "operation" in sheet_lower) or ("income" in sheet_lower and "statement" in sheet_lower) or ("statement" in sheet_lower and "earnings" in sheet_lower):
            is_sheet = sheet
            
    if not bs_sheet:
        print("Could not identify Balance Sheet!")
    if not is_sheet:
        print("Could not identify Income Statement!")
        
    print(f"Selected Balance Sheet: {bs_sheet}")
    print(f"Selected Income Statement: {is_sheet}\n")
    
    # Print first 15 rows of each
    if bs_sheet:
        df_bs = pd.read_excel(file_path, sheet_name=bs_sheet)
        print(f"\n--- {bs_sheet} (First 15 Rows) ---")
        print(df_bs.head(15).to_string())
        print("Columns:", df_bs.columns.tolist())
        
    if is_sheet:
        df_is = pd.read_excel(file_path, sheet_name=is_sheet)
        print(f"\n--- {is_sheet} (First 15 Rows) ---")
        print(df_is.head(15).to_string())
        print("Columns:", df_is.columns.tolist())

if __name__ == "__main__":
    file1 = "data/1070985_2021_10K.xlsx"
    file2 = "data/1534701_2023_10K.xlsx"
    
    explore_excel(file1)
    explore_excel(file2)
