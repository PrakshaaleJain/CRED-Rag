import os
import glob
import pandas as pd
import json
import logging
from pathlib import Path

def identify_sheets(filepath):
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    
    bs_sheet = None
    is_sheet = None
    
    for sheet in sheet_names:
        sheet_lower = sheet.lower()
        
        # Balance Sheet matcher
        if any(kw in sheet_lower for kw in ["balance sheet", "consolidated balance", "financial position"]):
            if "parenthetical" not in sheet_lower:
                bs_sheet = sheet
                
        # Income Statement matcher
        if any(kw in sheet_lower for kw in ["statement of operations", "statements of oper", "statement of oper", "statement of income", "statement of earnings"]):
            if "parenthetical" not in sheet_lower and "comprehensive" not in sheet_lower:
                is_sheet = sheet
                
    return bs_sheet, is_sheet

def extract_value(df, keywords, col_idx):
    if df is None or df.empty or col_idx >= len(df.columns):
        return None
        
    for idx, row in df.iterrows():
        label = str(row.iloc[0]).lower().strip()
        for kw in keywords:
            if kw.lower() in label:
                val = row.iloc[col_idx]
                if pd.isna(val) or str(val).strip() in ['-', '', 'None', 'nan']:
                    continue
                
                val_str = str(val).strip().replace('$', '').replace(',', '')
                if val_str.startswith('(') and val_str.endswith(')'):
                    val_str = '-' + val_str[1:-1]
                
                try:
                    return float(val_str)
                except ValueError:
                    pass
    return None

def extract_debt(df, col_idx):
    lt_debt = extract_value(df, ["long-term debt"], col_idx)
    st_debt = extract_value(df, ["short-term debt", "current portion of long-term debt"], col_idx)
    
    if lt_debt is None and st_debt is None:
        return None
    return (lt_debt or 0.0) + (st_debt or 0.0)

def safe_div(num, den):
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 4)

def safe_add(*args):
    if any(a is None for a in args):
        return None
    return sum(args)

def process_file(filepath):
    basename = os.path.basename(filepath)
    parts = basename.replace('.xlsx', '').split('_')
    
    cik = parts[0]
    try:
        filing_year = int(parts[1])
    except:
        filing_year = 0
        
    bs_sheet, is_sheet = identify_sheets(filepath)
    
    missing_critical = []
    if not bs_sheet: missing_critical.append("Balance Sheet")
    if not is_sheet: missing_critical.append("Income Statement")
    
    if missing_critical:
        return None, missing_critical
        
    try:
        df_bs = pd.read_excel(filepath, sheet_name=bs_sheet).dropna(how='all', axis=1)
        df_is = pd.read_excel(filepath, sheet_name=is_sheet).dropna(how='all', axis=1)
    except Exception as e:
        return None, [f"Excel Read Error: {str(e)}"]
        
    max_cols = min(4, len(df_bs.columns), len(df_is.columns))
    results = []
    
    for col_idx in range(1, max_cols):
        year_val = filing_year - (col_idx - 1)
        
        total_assets = extract_value(df_bs, ["total assets"], col_idx)
        total_equity = extract_value(df_bs, ["total equity", "total stockholders' equity", "stockholders' equity"], col_idx)
        
        total_liab = extract_value(df_bs, ["total liabilities"], col_idx)
        if total_liab is None and total_assets is not None and total_equity is not None:
            total_liab = total_assets - total_equity
            
        current_assets = extract_value(df_bs, ["total current assets"], col_idx)
        current_liab = extract_value(df_bs, ["total current liabilities"], col_idx)
        inventory = extract_value(df_bs, ["inventories", "inventory"], col_idx)
        retained_earnings = extract_value(df_bs, ["retained earnings", "accumulated deficit"], col_idx)
        
        total_debt = extract_debt(df_bs, col_idx)
        
        total_revenue = extract_value(df_is, ["total revenues", "total revenues and other income", "sales and other operating revenues"], col_idx)
        net_income = extract_value(df_is, ["net income", "net income (loss)", "net income attributable to"], col_idx)
        interest_exp = extract_value(df_is, ["interest expense", "interest and debt expense"], col_idx)
        da = extract_value(df_is, ["depreciation and amortization"], col_idx)
        taxes = extract_value(df_is, ["income tax expense", "provision for income taxes", "income taxes"], col_idx)
        
        if total_assets is None and net_income is None:
            continue
            
        ebit = safe_add(net_income, taxes, interest_exp)
        ebitda = safe_add(ebit, da)
        ffo = safe_add(net_income, da)
        
        interest_coverage = safe_div(ebitda, interest_exp)
        dscr = safe_div(ffo, total_debt)
        debt_to_equity = safe_div(total_debt, total_equity)
        re_to_ta = safe_div(retained_earnings, total_assets)
        current_ratio = safe_div(current_assets, current_liab)
        
        quick_ratio = None
        if current_assets is not None and inventory is not None and current_liab is not None:
            quick_ratio = safe_div(current_assets - inventory, current_liab)
            
        wc_to_ta = None
        if current_assets is not None and current_liab is not None and total_assets is not None:
            wc_to_ta = safe_div(current_assets - current_liab, total_assets)
            
        roce = safe_div(ebit, total_assets)
        npm = safe_div(net_income, total_revenue)
        tl_to_ta = safe_div(total_liab, total_assets)
        
        results.append({
            "File_Name": basename,
            "CIK_Identifier": cik,
            "Fiscal_Year": year_val,
            "Interest Coverage Ratio": interest_coverage,
            "DSCR": dscr,
            "Debt-to-Equity": debt_to_equity,
            "Retained Earnings / Total Assets": re_to_ta,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Working Capital / Total Assets": wc_to_ta,
            "ROCE": roce,
            "Net Profit Margin": npm,
            "Total Liabilities / Total Assets": tl_to_ta
        })
        
    return results, None

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    target_dirs = [
        project_root / 'data',
        project_root / 'data' / 'KPI_tables',
        project_root / 'data' / 'KPI_tables_cold_start'
    ]
    
    all_files = []
    for d in target_dirs:
        if d.exists():
            all_files.extend(list(d.glob("*.xlsx")))
            
    # Ignore temp files
    all_files = [f for f in all_files if not f.name.startswith("~$")]
    
    master_results = []
    audit_log = []
    
    audit_log.append("=== EXTRACTION REPORT ===")
    audit_log.append(f"Total .xlsx files discovered: {len(all_files)}\n")
    
    success_count = 0
    failed_count = 0
    
    for filepath in all_files:
        try:
            results, errors = process_file(filepath)
            if errors:
                failed_count += 1
                audit_log.append(f"[FAILED] {filepath.name}: Missing -> {', '.join(errors)}")
            elif not results:
                failed_count += 1
                audit_log.append(f"[FAILED] {filepath.name}: No yearly data found.")
            else:
                success_count += 1
                master_results.extend(results)
        except Exception as e:
            failed_count += 1
            audit_log.append(f"[ERROR] {filepath.name}: Exception -> {str(e)}")
            
    audit_log.insert(2, f"Successfully processed: {success_count}")
    audit_log.insert(3, f"Failed/Skipped: {failed_count}\n")
    
    # Save outputs
    df_master = pd.DataFrame(master_results)
    
    csv_out = project_root / 'data' / 'credit_risk_kpis_master.csv'
    json_out = project_root / 'data' / 'credit_risk_kpis_master.json'
    report_out = project_root / 'data' / 'extraction_report.txt'
    
    df_master.to_csv(csv_out, index=False)
    
    # Save JSON hierarchical
    df_master.to_json(json_out, orient='records', indent=4)
    
    # Save audit log
    with open(report_out, 'w') as f:
        f.write('\n'.join(audit_log))
        
    print(f"Success! Master dataset generated.")
    print(f"Saved CSV: {csv_out}")
    print(f"Saved JSON: {json_out}")
    print(f"Saved Report: {report_out}")
    
    print("\nSample Markdown Table:")
    print(df_master.head(10).to_markdown(index=False))

if __name__ == '__main__':
    main()
