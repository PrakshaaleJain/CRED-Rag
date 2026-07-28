#!/usr/bin/env python3
"""Batch process HTML SEC filings into JSON extracted narratives."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure we can import from utils
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.html_item_extractor import extract_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    parser = argparse.ArgumentParser(description="Batch extract narrative items from SEC HTML filings.")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=project_root / "data" / "egan_sec_filings_html",
        help="Directory containing the HTML filings"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=project_root / "data" / "egan_sec_filings_extracted_text", 
        help="Directory to save the JSONs"
    )
    parser.add_argument(
        "--items", 
        type=str, 
        default="1,1A,3,7,10,11", 
        help="Comma-separated items to extract"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    html_files = list(args.input_dir.glob("*.htm")) + list(args.input_dir.glob("*.html"))
    
    if not html_files:
        logging.error(f"No .htm or .html files found in {args.input_dir}")
        return

    items_list = [x.strip() for x in args.items.split(',') if x.strip()]
    total = len(html_files)

    for idx, html_path in enumerate(html_files, start=1):
        out_path = args.output_dir / f"{html_path.stem}_extracted.json"
        
        logging.info(f"[{idx}/{total}] Extracting {html_path.name}...")
        try:
            res = extract_items(str(html_path), items_list)
            out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to extract {html_path.name}: {e}")

    logging.info("Batch extraction complete!")

if __name__ == "__main__":
    main()
