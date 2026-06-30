"""
SEC 10-K Filing Scraper for CRED-RAPTOR
========================================
Simple script to fetch ONE 10-K filing from SEC EDGAR and save it locally.
Run this first to understand the structure before scaling up.
"""

from dotenv import load_dotenv
import requests
import pandas as pd
import os
import re
from io import StringIO
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

load_dotenv()
email = os.getenv("scraper_email")  # Load email from .env file for User-Agent header


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# SEC requires a User-Agent header identifying who is making requests
# Replace with your actual email
# ─────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": email}

# Where to save the downloaded filing on your laptop
SAVE_FOLDER = "sec_filings"
SAVE_md_FOLDER = "sec_filings_md"
os.makedirs(SAVE_FOLDER, exist_ok=True)
sample_number = 3


def html_to_markdown_with_json_tables(html: str) -> str:
    """Convert filing HTML into markdown text and preserve tables as markdown tables."""
    head = html.lstrip()[:300].lower()
    is_xml_like = head.startswith("<?xml") or "<xbrl" in head or "ix:" in head

    if is_xml_like:
        try:
            soup = BeautifulSoup(html, "lxml-xml")
        except Exception:
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "xml")
    else:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # Preserve headings as markdown headings
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            text = " ".join(h.stripped_strings)
            h.replace_with(f"\n{'#' * level} {text}\n")

    # Preserve list items in markdown style
    for li in soup.find_all("li"):
        text = " ".join(li.stripped_strings)
        li.replace_with(f"\n- {text}\n")

    # Preserve paragraph breaks
    for p in soup.find_all("p"):
        text = " ".join(p.stripped_strings)
        p.replace_with(f"\n{text}\n")

    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Convert each table to markdown (kept in original order)
    for table in soup.find_all("table"):
        table_md = ""
        try:
            frames = pd.read_html(StringIO(str(table)), flavor="lxml")
            if frames:
                df = frames[0].fillna("")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [" ".join(str(c) for c in col if str(c) != "nan").strip() for col in df.columns]
                else:
                    df.columns = [str(c).strip() for c in df.columns]
                table_md = df.to_markdown(index=False)
        except Exception:
            rows = []
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                row = [" ".join(cell.stripped_strings) for cell in cells]
                if row:
                    rows.append(row)
            if rows:
                max_cols = max(len(r) for r in rows)
                rows = [r + [""] * (max_cols - len(r)) for r in rows]
                if len(rows) > 1:
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                else:
                    df = pd.DataFrame(rows)
                table_md = df.to_markdown(index=False)

        table.replace_with(f"\n\n{table_md}\n\n" if table_md else "\n")

    container = soup.body if soup.body else soup
    text = container.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Get the master list of all companies registered with SEC
# This JSON file has every company's: name, ticker, CIK number
# CIK (Central Index Key) = unique ID SEC assigns to each company
# ─────────────────────────────────────────────────────────────────
print("Step 1: Fetching master list of all SEC-registered companies...")
#response is the json variable which contains the companies cik and its name
response = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=HEADERS
)

# Convert the JSON response into a readable DataFrame
companies = pd.DataFrame.from_dict(response.json(), orient='index')
companies['cik_str'] = companies['cik_str'].astype(str).str.zfill(10)

print(f"  → Found {len(companies)} companies.\n")
# print(companies.head(5))  # 

 

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Pick sample_number companies to work with
# Randomly sample a few companies so multiple filings are saved
# ─────────────────────────────────────────────────────────────────
sample_n = min(sample_number, len(companies))
sampled_companies = companies.sample(n=sample_n, random_state=None).reset_index(drop=True)

print(f"\nStep 2: Selected {len(sampled_companies)} companies.")

saved_files = []
failed_companies = []

for idx, company_row in sampled_companies.iterrows():
    cik = company_row['cik_str']
    ticker = company_row['ticker']
    name = company_row['title']

    print(f"\nFetching filing history for {name} ({ticker})...")

    submissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=HEADERS
    )

    if submissions.status_code != 200:
        print(f"  → Failed to fetch submissions. HTTP status: {submissions.status_code}")
        failed_companies.append((ticker, name, "submissions request failed"))
        continue

    # The actual filings list is nested under filings → recent
    recent_filings = submissions.json()['filings']['recent']

    # Convert to DataFrame so we can filter easily
    all_filings = pd.DataFrame.from_dict(recent_filings)

    # print(f"  → Total filings found: {len(all_filings)}")
    # print(f"  → Filing types available: {all_filings['form'].unique()[:10]}")  # preview types

    # ─────────────────────────────────────────────────────────────────
    # STEP 4 — Filter to get only the 10-K (annual report) filings
    # ─────────────────────────────────────────────────────────────────
    ten_k_filings = all_filings[all_filings['form'] == '10-K'].reset_index(drop=True)

    # print(f"\nStep 4: Filtering for 10-K filings only...")
    # print(f"  → Found {len(ten_k_filings)} 10-K filings for {name}")

    if ten_k_filings.empty:
        print("  → Skipping: no 10-K filings found")
        failed_companies.append((ticker, name, "no 10-K filing"))
        continue

    # ─────────────────────────────────────────────────────────────────
    # STEP 5 — Pick the MOST RECENT 10-K filing and download it
    # ─────────────────────────────────────────────────────────────────
    # print(f"\nStep 5: Picking the most recent 10-K filing...")

    latest = ten_k_filings.iloc[0]  # first row = most recent

    accession_number = latest['accessionNumber']
    filing_date      = latest['filingDate']
    primary_doc      = latest['primaryDocument']

    print(f"  → Filing date    : {filing_date}")
    print(f"  → Accession No.  : {accession_number}")
    print(f"  → Primary doc    : {primary_doc}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6 — Build the download URL and fetch the filing text
    # print(f"\nStep 6: Downloading the 10-K filing text...")

    accession_clean = accession_number.replace("-", "")
    cik_no_zeros = str(int(cik))

    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_clean}/{primary_doc}"
    )
    print(f"  → URL: {filing_url}")

    filing_response = requests.get(filing_url, headers=HEADERS)

    if filing_response.status_code == 200:
        print(f"  → Downloaded successfully! Size: {len(filing_response.text):,} characters")
    else:
        print(f"  → Failed to download. HTTP status: {filing_response.status_code}")
        failed_companies.append((ticker, name, "filing download failed"))
        continue

    # ─────────────────────────────────────────────────────────────────
    # STEP 7 — Save the filing in markdown format
    # ─────────────────────────────────────────────────────────────────
    year      = filing_date[:4]
    save_name = f"{ticker}_{year}_10K.md"
    save_htm_name = f"{ticker}_{year}_10K.htm"
    save_path = os.path.join(SAVE_md_FOLDER, save_name)
    save_htm_path = os.path.join(SAVE_FOLDER, save_htm_name)

    markdown_text = html_to_markdown_with_json_tables(filing_response.text)

    with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(markdown_text)

    with open(save_htm_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(filing_response.text)

    saved_files.append(save_path)
    # print(f"\nStep 7: Filing saved!")
    print(f"  → Saved to : {save_path}")
    print(f"  → Open the .htm file in your browser to read the full annual report.")


print(f"\nDone! Saved {len(saved_files)} filing(s) to '{SAVE_FOLDER}'.")
if failed_companies:
    print(f"  → Skipped {len(failed_companies)} company(ies)")