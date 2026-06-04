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
os.makedirs(SAVE_FOLDER, exist_ok=True)


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

# CIK must be 10 digits with leading zeros (SEC API requirement)
companies['cik_str'] = companies['cik_str'].astype(str).str.zfill(10)

print(f"  → Found {len(companies)} companies.\n")
print(companies.head(5))  # preview first 5 rows
 

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Pick ONE company to work with
# We are using Apple (AAPL) as an example
# You can change the ticker below to any company you want
# ─────────────────────────────────────────────────────────────────
TARGET_TICKER = "AAPL"

# Find the row where ticker matches our target
company_row = companies[companies['ticker'] == TARGET_TICKER].iloc[0]
cik         = company_row['cik_str']       # e.g. "0000320193"
name        = company_row['title']         # e.g. "Apple Inc."

print(f"\nStep 2: Selected company → {name} | Ticker: {TARGET_TICKER} | CIK: {cik}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Fetch the filing history for this company
# The submissions endpoint returns metadata for ALL filings ever made
# by this company (10-K, 10-Q, 8-K, etc.)
# ─────────────────────────────────────────────────────────────────
print(f"\nStep 3: Fetching filing history for {name}...")

submissions = requests.get(
    f"https://data.sec.gov/submissions/CIK{cik}.json",
    headers=HEADERS
)

# The actual filings list is nested under filings → recent
recent_filings = submissions.json()['filings']['recent']

# Convert to DataFrame so we can filter easily
all_filings = pd.DataFrame.from_dict(recent_filings)

print(f"  → Total filings found: {len(all_filings)}")
print(f"  → Filing types available: {all_filings['form'].unique()[:10]}")  # preview types


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Filter to get only the 10-K (annual report) filings
# 10-K is the full annual report — the document RAPTOR will process
# It contains: business overview, risk factors, financials, MD&A
# ─────────────────────────────────────────────────────────────────
ten_k_filings = all_filings[all_filings['form'] == '10-K'].reset_index(drop=True)

print(f"\nStep 4: Filtering for 10-K filings only...")
print(f"  → Found {len(ten_k_filings)} 10-K filings for {name}")
print(ten_k_filings[['accessionNumber', 'filingDate', 'primaryDocument']].head(10))

#################### WORKING TILL HERE ###################################
## Now we have a list of 10K filings with the metadata for a single company.

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Pick the MOST RECENT 10-K filing and download it
# accessionNumber = unique ID for this specific filing submission
# primaryDocument = the main file inside that submission folder
# ─────────────────────────────────────────────────────────────────
print(f"\nStep 5: Picking the most recent 10-K filing...")

latest = ten_k_filings.iloc[0]  # first row = most recent

accession_number = latest['accessionNumber']          # e.g. "0000320193-24-000123"
filing_date      = latest['filingDate']               # e.g. "2024-11-01"
primary_doc      = latest['primaryDocument']          # e.g. "aapl-20240928.htm"

print(f"  → Filing date    : {filing_date}")
print(f"  → Accession No.  : {accession_number}")
print(f"  → Primary doc    : {primary_doc}")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — Build the download URL and fetch the filing text
#
# SEC stores filings at:
# https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/{PRIMARY_DOC}
#
# The accession number has dashes when shown as text (e.g. 0000320193-24-000123)
# but the folder path uses it WITHOUT dashes (e.g. 000032019324000123)
# ─────────────────────────────────────────────────────────────────
print(f"\nStep 6: Downloading the 10-K filing text...")

# Remove dashes from accession number for the URL path
accession_clean = accession_number.replace("-", "")

# CIK without leading zeros for the /data/{CIK}/ part of the path
cik_no_zeros = str(int(cik))

# Build the full URL
filing_url = (
    f"https://www.sec.gov/Archives/edgar/data/"
    f"{cik_no_zeros}/{accession_clean}/{primary_doc}"
)
print(f"  → URL: {filing_url}")

# Download the filing
filing_response = requests.get(filing_url, headers=HEADERS)

if filing_response.status_code == 200:
    print(f"  → Downloaded successfully! Size: {len(filing_response.text):,} characters")
else:
    print(f"  → Failed to download. HTTP status: {filing_response.status_code}")
    exit()


# ─────────────────────────────────────────────────────────────────
# STEP 7 — Save the raw filing text to your laptop
# The file is saved as .htm (HTML format) — you can open it in
# any browser to read it, or process it as text in Python
# ─────────────────────────────────────────────────────────────────
year      = filing_date[:4]                          # e.g. "2024"
save_name = f"{TARGET_TICKER}_{year}_10K.htm"        # e.g. "AAPL_2024_10K.htm"
save_path = os.path.join(SAVE_FOLDER, save_name)

with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
    f.write(filing_response.text)

print(f"\nStep 7: Filing saved!")
print(f"  → Saved to : {save_path}")
print(f"  → Open the .htm file in your browser to read the full annual report.")


# ─────────────────────────────────────────────────────────────────
# STEP 8 — Quick preview of what's inside the filing
# Print the first 3000 characters so you can see the raw content
# ─────────────────────────────────────────────────────────────────
print(f"\nStep 8: Preview of first 3000 characters of the filing:\n")
print("-" * 60)
print(filing_response.text[:3000])
print("-" * 60)
print("\nDone! Check your 'sec_filings' folder for the saved file.")