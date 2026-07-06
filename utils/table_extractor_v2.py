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
OUTPUT_DIR = Path(os.getenv("_tables"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

############################################################
# HELPER
############################################################

def html_table_to_dataframe(table):

    rows = []

    for tr in table.find_all("tr"):

        row = []

        for cell in tr.find_all(["th", "td"]):
            row.append(cell.get_text(" ", strip=True))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    return pd.DataFrame(rows)

############################################################
# PROCESS FILINGS
############################################################

html_files = sorted(
    f for f in HTML_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in {".html", ".htm"}
)

for html_file in html_files:

    print(f"Processing {html_file.name}")

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    tables = soup.find_all("table")

    output_excel = OUTPUT_DIR / f"{html_file.stem}.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

        current_row = 0

        for idx, table in enumerate(tables, start=1):

            df = html_table_to_dataframe(table)

            if df.empty:
                continue

            # Write title
            title = pd.DataFrame([[f"TABLE {idx}"]])

            title.to_excel(
                writer,
                sheet_name="Tables",
                startrow=current_row,
                index=False,
                header=False
            )

            current_row += 2

            # Write table
            df.to_excel(
                writer,
                sheet_name="Tables",
                startrow=current_row,
                index=False,
                header=False
            )

            current_row += len(df) + 4

    print(f"Saved -> {output_excel}")

print("Done.")