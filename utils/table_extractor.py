"""
SEC 10-K HTML Table Extractor
=============================

Extracts every HTML <table> from every SEC filing and saves each table
as an individual CSV file.

Output:

data/
└── extracted_tables/
    ├── HAVA_2026_10K/
    │     001.csv
    │     002.csv
    │     ...
    │
    └── ZETA_2026_10K/
          001.csv
          002.csv
          ...

Requirements:
pip install beautifulsoup4 pandas lxml python-dotenv
"""

from pathlib import Path
import os

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv


############################################################
# CONFIG
############################################################

load_dotenv()

HTML_DIR = Path(os.getenv("sec_filings_html"))
OUTPUT_DIR = Path(os.getenv("sec_tables_db"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

############################################################
# FIND ALL HTML FILES
############################################################

html_files = sorted(
    f for f in HTML_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in {".htm", ".html"}
)

print(f"Found {len(html_files)} SEC filings.\n")

############################################################
# HELPER
############################################################


def html_table_to_dataframe(table):
    rows = []
    for tr in table.find_all("tr"):

        row = []

        for cell in tr.find_all(["th", "td"]):
            text = cell.get_text(" ", strip=True)
            row.append(text)

        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)

    normalized_rows = []

    for row in rows:

        row = row + [""] * (max_cols - len(row))

        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows)


############################################################
# PROCESS FILINGS
############################################################

total_tables = 0

for html_file in html_files:

    print("=" * 80)
    print(f"Processing: {html_file.name}")

    company_name = html_file.stem

    company_output_dir = OUTPUT_DIR / company_name

    company_output_dir.mkdir(parents=True, exist_ok=True)

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    tables = soup.find_all("table")

    print(f"Found {len(tables)} tables")

    total_tables += len(tables)

    saved = 0

    for idx, table in enumerate(tables, start=1):

        df = html_table_to_dataframe(table)

        # Skip completely empty tables
        if df.empty:
            continue

        csv_path = company_output_dir / f"{idx:03d}.csv"

        df.to_csv(csv_path, index=False, header=False)

        saved += 1

    print(f"Saved {saved} CSV files.")

print("\n" + "=" * 80)
print(f"Processed {len(html_files)} filings")
print(f"Extracted {total_tables} tables")
print(f"Output directory: {OUTPUT_DIR.resolve()}")