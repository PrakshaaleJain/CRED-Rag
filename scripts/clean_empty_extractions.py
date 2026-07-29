#!/usr/bin/env python3
"""
Script to scan the extracted JSON narratives and delete files that contain
'junk' or failed extractions (e.g., where the script only grabbed the Table of Contents headers).
"""

import json
import logging
from pathlib import Path

# Minimum total character count to consider an extraction "valid"
# A real 10-K MD&A (Item 7) and Risk Factors (Item 1A) will easily exceed 10,000 characters.
# Setting a conservative threshold of 2,000 characters to filter out just the headers.
MIN_CHARS_THRESHOLD = 2000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    project_root = Path(__file__).resolve().parents[1]
    extracted_dir = project_root / "data" / "egan_sec_filings_extracted_text"
    
    if not extracted_dir.exists():
        logging.error(f"Directory not found: {extracted_dir}")
        return

    json_files = list(extracted_dir.glob("*.json"))
    logging.info(f"Scanning {len(json_files)} extracted files for junk...")

    deleted_count = 0
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            total_chars = 0
            # Sum up the length of the text in all extracted items
            for item_key, content in data.items():
                text = content.get("text", "")
                total_chars += len(text)
                
            # If the total extracted text is tiny, it's a failed extraction (Table of Contents junk)
            if total_chars < MIN_CHARS_THRESHOLD:
                logging.info(f"Deleting {json_path.name} (Only {total_chars} characters extracted)")
                json_path.unlink()  # Delete the file
                deleted_count += 1
                
        except Exception as e:
            logging.error(f"Error processing {json_path.name}: {e}")
            
    logging.info(f"Cleanup complete! Deleted {deleted_count} junk files out of {len(json_files)}.")
    logging.info(f"Remaining valid files: {len(json_files) - deleted_count}")

if __name__ == "__main__":
    main()
