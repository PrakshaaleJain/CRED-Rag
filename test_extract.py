import pandas as pd
import re

TARGET_ITEMS = {
    'Total Assets': r'^(Total\s+Assets|Assets)$',
    'Net Income': r'^(Net\s+Income\s+\(Loss\)|Net\s+Income|Net\s+Loss|Net\s+income\s+\(loss\)\s+attributable\s+to.*)$',
}

def clean_number(val):
    if pd.isna(val): return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if s in ['—', '-', '', ')']: return 0.0
    is_negative = False
    if '(' in s or ')' in s:
        is_negative = True
        s = s.replace('(', '').replace(')', '')
    s = s.replace(',', '').replace('$', '').strip()
    try:
        num = float(s)
        return -num if is_negative else num
    except ValueError:
        return None

tables = pd.read_html('data/sec_filings_html/HAVA_2026_10K.htm')
found = {}
for df in tables:
    df = pd.DataFrame(df.values) # Convert to integer columns
    for index, row in df.iterrows():
        first_col_val = str(row[0]).strip() if len(row) > 0 and not pd.isna(row[0]) else ""
        if not first_col_val:
            first_col_val = str(row[1]).strip() if len(row) > 1 and not pd.isna(row[1]) else ""
        if not first_col_val: continue
        label = re.sub(r'[^a-zA-Z0-9\s\(\)\-]', '', first_col_val).strip()
        for item_name, pattern in TARGET_ITEMS.items():
            if re.match(pattern, label, re.IGNORECASE):
                nums = [clean_number(row[i]) for i in range(1, len(row)) if clean_number(row[i]) is not None]
                if nums and item_name not in found:
                    found[item_name] = nums
print(found)
