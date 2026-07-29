#!/usr/bin/env python3
"""Batch process HTML SEC filings into JSON extracted narratives using multiprocessing."""

import argparse
import json
import logging
import sys
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure we can import from utils
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.html_item_extractor import extract_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_file(html_path, output_dir, items_list):
    """Worker function to process a single file."""
    out_path = output_dir / f"{html_path.stem}_extracted.json"
    
    try:
        res = extract_items(str(html_path), items_list)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
        return html_path.name, True, None
    except Exception as e:
        return html_path.name, False, str(e)

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
        default="1,1A,3,7,7A,15", 
        help="Comma-separated items to extract"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel worker processes to use (defaults to all CPU cores)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    html_files = list(args.input_dir.glob("*.htm")) + list(args.input_dir.glob("*.html"))
    
    if not html_files:
        logging.error(f"No .htm or .html files found in {args.input_dir}")
        return

    items_list = [x.strip() for x in args.items.split(',') if x.strip()]
    total = len(html_files)
    
    logging.info(f"Starting extraction of {total} files using {args.workers} parallel workers...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks to the process pool
        futures = {
            executor.submit(process_file, html_path, args.output_dir, items_list): html_path 
            for html_path in html_files
        }
        
        # Log progress as soon as each file finishes extracting
        for i, future in enumerate(as_completed(futures), start=1):
            filename, success, error_msg = future.result()
            
            if success:
                success_count += 1
                logging.info(f"[{i}/{total}] Successfully extracted {filename}")
            else:
                logging.error(f"[{i}/{total}] Failed to extract {filename}: {error_msg}")

    logging.info(f"Batch extraction complete! Successfully processed {success_count}/{total} files.")

if __name__ == "__main__":
    main()
