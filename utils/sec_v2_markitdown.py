"""
SEC 10-K Filing Scraper for CRED-RAPTOR
======================================
Downloads the latest 10-K filing for a random sample of SEC companies,
saves the original HTML, and converts it to Markdown using Microsoft's
MarkItDown library.
"""

from dotenv import load_dotenv
from markitdown import MarkItDown
import requests
import pandas as pd
import os

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

email = os.getenv("scraper_email")

if not email:
    raise ValueError(
        "Please set 'scraper_email' in your .env file.\n"
        "Example:\n"
        "scraper_email=your_email@example.com"
    )

HEADERS = {
    "User-Agent": email
}

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
SAVE_HTML_FOLDER = "sec_filings"
SAVE_MD_FOLDER = "sec_filings_md"

os.makedirs(SAVE_HTML_FOLDER, exist_ok=True)
os.makedirs(SAVE_MD_FOLDER, exist_ok=True)

sample_number = 3 #10 number of companies to sample for 10-K filings
md_converter = MarkItDown()

# -------------------------------------------------------------------
# STEP 1 — Fetch SEC company master list
# -------------------------------------------------------------------
print("Step 1: Fetching SEC company master list...")

response = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=HEADERS,
)

response.raise_for_status()

companies = pd.DataFrame.from_dict(response.json(), orient="index")
companies["cik_str"] = companies["cik_str"].astype(str).str.zfill(10)

print(f"  → Found {len(companies)} companies.\n")

# -------------------------------------------------------------------
# STEP 2 — Randomly sample companies
# -------------------------------------------------------------------
sample_n = min(sample_number, len(companies))
sampled_companies = (
    companies.sample(n=sample_n)
    .reset_index(drop=True)
)

print(f"Step 2: Selected {len(sampled_companies)} companies.")

saved_files = []
failed_companies = []

# -------------------------------------------------------------------
# STEP 3 onwards
# -------------------------------------------------------------------
for _, company in sampled_companies.iterrows():

    cik = company["cik_str"]
    ticker = company["ticker"]
    company_name = company["title"]

    print(f"\n{'=' * 70}")
    print(f"{company_name} ({ticker})")
    print("=" * 70)

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    submissions = requests.get(
        submissions_url,
        headers=HEADERS,
    )

    if submissions.status_code != 200:
        print(f"Failed to fetch submissions ({submissions.status_code})")
        failed_companies.append((ticker, company_name))
        continue

    recent = submissions.json()["filings"]["recent"]
    filings = pd.DataFrame(recent)

    ten_k = filings[filings["form"] == "10-K"].reset_index(drop=True)

    if ten_k.empty:
        print("No 10-K filings found.")
        failed_companies.append((ticker, company_name))
        continue

    latest = ten_k.iloc[0]

    accession = latest["accessionNumber"]
    filing_date = latest["filingDate"]
    primary_doc = latest["primaryDocument"]

    accession_clean = accession.replace("-", "")
    cik_no_zero = str(int(cik))

    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zero}/{accession_clean}/{primary_doc}"
    )

    print(f"Filing Date : {filing_date}")
    print(f"Downloading : {filing_url}")

    filing = requests.get(
        filing_url,
        headers=HEADERS,
    )

    if filing.status_code != 200:
        print(f"Download failed ({filing.status_code})")
        failed_companies.append((ticker, company_name))
        continue

    year = filing_date[:4]

    html_filename = f"{ticker}_{year}_10K.htm"
    md_filename = f"{ticker}_{year}_10K.md"

    html_path = os.path.join(
        SAVE_HTML_FOLDER,
        html_filename,
    )

    md_path = os.path.join(
        SAVE_MD_FOLDER,
        md_filename,
    )

    # ---------------------------------------------------------------
    # Save HTML
    # ---------------------------------------------------------------
    with open(html_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(filing.text)

    # ---------------------------------------------------------------
    # Convert HTML -> Markdown using MarkItDown
    # ---------------------------------------------------------------
    try:
        result = md_converter.convert(html_path)
        with open( md_path,"w", encoding="utf-8",errors="ignore") as f:
            f.write(result.text_content)

        print(f"HTML saved      : {html_path}")
        print(f"Markdown saved  : {md_path}")

        saved_files.append(md_path)

    except Exception as e:
        print(f"Markdown conversion failed: {e}")
        failed_companies.append((ticker, company_name))

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("Finished")
print("=" * 70)

print(f"Successfully saved : {len(saved_files)} filings")

if failed_companies:
    print(f"Failed             : {len(failed_companies)} companies")
    for ticker, company in failed_companies:
        print(f"  - {ticker}: {company}")
else:
    print("No failures.")