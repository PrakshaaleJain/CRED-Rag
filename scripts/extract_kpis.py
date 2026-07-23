import os
import json
import re
import math
import pandas as pd
import numpy as np

# Define mappings from standard names to regex patterns
# The patterns handle common variations in SEC filings
TARGET_ITEMS = {
    'Total Assets': r'^(Total\s+Assets|Assets)$',
    'Current Assets': r'^(Total\s+Current\s+Assets|Current\s+Assets)$',
    'Total Liabilities': r'^(Total\s+Liabilities|Liabilities)$',
    'Current Liabilities': r'^(Total\s+Current\s+Liabilities|Current\s+Liabilities)$',
    'Total Equity': r'^(Total\s+Stockholders(\'|\’|)\s+Equity|Total\s+Shareholders(\'|\’|)\s+(Equity|Deficit)|Stockholders(\'|\’|)\s+Equity|Total\s+Liabilities\s+and\s+Stockholders(\'|\’|)\s+Equity\s+\-\s+Total\s+Liabilities)$',
    'Retained Earnings': r'^(Retained\s+Earnings|Accumulated\s+Deficit|Retained\s+Earnings\s+\(Accumulated\s+Deficit\))$',
    'Net Income': r'^(Net\s+Income\s+\(Loss\)|Net\s+Income|Net\s+Loss|Net\s+income\s+\(loss\)\s+attributable\s+to.*)$',
    'Total Revenue': r'^(Total\s+Revenues|Revenues|Total\s+Revenue|Total\s+net\s+revenues|Revenue|Sales|Net\s+Sales)$',
    'EBIT': r'^(Operating\s+Income\s+\(Loss\)|Operating\s+Income|Operating\s+Loss|Income\s+\(loss\)\s+from\s+operations)$',
    'Interest Expense': r'^(Interest\s+Expense|Interest\s+expense,\s+net|Interest\s+expense)$',
    'Operating Cash Flow': r'^(Net\s+cash\s+provided\s+by\s+\(used\s+in\)\s+operating\s+activities|Net\s+cash\s+provided\s+by\s+operating\s+activities|Net\s+Cash\s+used\s+in\s+Operating\s+Activities)$',
    'Cash & Short-Term Investments': r'^(Cash\s+and\s+cash\s+equivalents|Cash\s+and\s+short-term\s+investments|Cash,\s+cash\s+equivalents\s+and\s+short-term\s+investments|Cash,\s+end\s+of\s+period|Cash)$'
}

def clean_number(val):
    if pd.isna(val):
        return None
    
    if isinstance(val, (int, float)):
        return float(val)
        
    s = str(val).strip()
    # Check for empty or non-numeric placeholder
    if s in ['—', '-', '', ')']:
        return 0.0
        
    # Handle parentheses for negative numbers
    is_negative = False
    if '(' in s or ')' in s:
        is_negative = True
        s = s.replace('(', '').replace(')', '')
        
    # Remove commas and $ signs
    s = s.replace(',', '').replace('$', '').strip()
    
    try:
        num = float(s)
        return -num if is_negative else num
    except ValueError:
        return None

def process_excel_file(filepath):
    print(f"Processing {filepath}...")
    df = pd.read_excel(filepath, sheet_name=0, header=None)
    
    extracted_data = {
        'Raw_Items': {},
        'KPIs': {}
    }
    
    # Initialize all raw items to None
    for item in TARGET_ITEMS.keys():
        extracted_data['Raw_Items'][item + '_t'] = None
        if item == 'Total Assets':
            extracted_data['Raw_Items'][item + '_t-1'] = None
            
    # Iterate over rows
    for index, row in df.iterrows():
        first_col_val = str(row[0]).strip() if not pd.isna(row[0]) else ""
        if not first_col_val:
            # Maybe the label is in the second column
            first_col_val = str(row[1]).strip() if len(row) > 1 and not pd.isna(row[1]) else ""
            
        if not first_col_val:
            continue
            
        # Clean up label for matching
        label = re.sub(r'[^a-zA-Z0-9\s\(\)\-]', '', first_col_val).strip()
        
        for item_name, pattern in TARGET_ITEMS.items():
            if re.match(pattern, label, re.IGNORECASE):
                # We found a match, extract numbers from the remaining columns
                numbers = []
                for col_idx in range(1, len(row)):
                    val = clean_number(row[col_idx])
                    if val is not None:
                        numbers.append(val)
                
                if numbers:
                    # Prefer the first match if already found (to avoid lower down notes replacing actual line items)
                    if extracted_data['Raw_Items'].get(item_name + '_t') is None:
                        extracted_data['Raw_Items'][item_name + '_t'] = numbers[0]
                        if item_name == 'Total Assets' and len(numbers) > 1:
                            extracted_data['Raw_Items'][item_name + '_t-1'] = numbers[1]
                break

    # Some fallback logic
    raw = extracted_data['Raw_Items']
    
    # Calculate KPIs
    # Safe division helper
    def safe_div(num, den):
        if num is None or den is None or den == 0:
            return None
        return num / den

    kpi = extracted_data['KPIs']
    
    # X1
    kpi['X1_Altman_Current_Assets_to_Total'] = safe_div(
        (raw['Current Assets_t'] or 0) - (raw['Current Liabilities_t'] or 0), 
        raw['Total Assets_t']
    )
    
    # X2
    kpi['X2_Altman_Retained_Earnings_to_Total'] = safe_div(
        raw['Retained Earnings_t'], 
        raw['Total Assets_t']
    )
    
    # X3
    kpi['X3_Altman_EBIT_to_Total'] = safe_div(
        raw['EBIT_t'], 
        raw['Total Assets_t']
    )
    
    # X4
    kpi['X4_Altman_Equity_to_Liabilities'] = safe_div(
        raw['Total Equity_t'], 
        raw['Total Liabilities_t']
    )
    
    # X5
    kpi['X5_Altman_Revenue_to_Total'] = safe_div(
        raw['Total Revenue_t'], 
        raw['Total Assets_t']
    )
    
    # X6
    kpi['X6_Ohlson_Liabilities_to_Total'] = safe_div(
        raw['Total Liabilities_t'], 
        raw['Total Assets_t']
    )
    
    # X7
    kpi['X7_Ohlson_Net_Income_to_Total'] = safe_div(
        raw['Net Income_t'], 
        raw['Total Assets_t']
    )
    
    # X8
    kpi['X8_Ohlson_Current_Assets_to_Current_Liabilities'] = safe_div(
        raw['Current Assets_t'], 
        raw['Current Liabilities_t']
    )
    
    # X9
    if raw['Total Assets_t'] and raw['Total Assets_t'] > 0:
        kpi['X9_Ohlson_Log_Total_Assets'] = math.log(raw['Total Assets_t'])
    else:
        kpi['X9_Ohlson_Log_Total_Assets'] = None
        
    # X10
    kpi['X10_Zmijewski_Operating_Cash_Flow_to_Liabilities'] = safe_div(
        raw['Operating Cash Flow_t'], 
        raw['Total Liabilities_t']
    )
    
    # X11
    kpi['X11_Zmijewski_Cash_to_Total_Assets'] = safe_div(
        raw['Cash & Short-Term Investments_t'], 
        raw['Total Assets_t']
    )
    
    # X12
    kpi['X12_Zmijewski_EBIT_to_Interest_Expense'] = safe_div(
        raw['EBIT_t'], 
        raw['Interest Expense_t']
    )
    
    # X13
    kpi['X13_Growth_Net_Income_to_Equity'] = safe_div(
        raw['Net Income_t'], 
        raw['Total Equity_t']
    )
    
    # X14
    kpi['X14_Growth_Liabilities_to_Equity'] = safe_div(
        raw['Total Liabilities_t'], 
        raw['Total Equity_t']
    )
    
    # X15
    kpi['X15_Growth_Assets_Growth'] = safe_div(
        (raw['Total Assets_t'] or 0) - (raw['Total Assets_t-1'] or 0),
        raw['Total Assets_t-1']
    )
    
    return extracted_data

if __name__ == "__main__":
    kpi_dir = "data/KPI_tables"
    for filename in os.listdir(kpi_dir):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(kpi_dir, filename)
            result = process_excel_file(filepath)
            
            output_name = filename.replace('.xlsx', '_KPIs.json')
            output_path = os.path.join(kpi_dir, output_name)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Saved KPIs to {output_path}")

