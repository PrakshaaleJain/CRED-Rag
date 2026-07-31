import os
import shutil
import csv
import logging
from pathlib import Path

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    project_root = Path(__file__).resolve().parents[1]
    
    labels_csv = project_root / 'data' / 'unrated_inference_set.csv'
    source_extracted_dir = project_root / 'data' / 'egan_sec_filings_extracted_text'
    
    final_extracted_dir = project_root / 'data' / 'final_sec_filings_extracted_text'
    inference_trees_dir = project_root / 'data' / 'inference_trees_directory'
    
    final_extracted_dir.mkdir(parents=True, exist_ok=True)
    inference_trees_dir.mkdir(parents=True, exist_ok=True)
    
    if not labels_csv.exists():
        logging.error(f"{labels_csv} not found. Please run scripts/create_inference_set.py first.")
        return
        
    logging.info(f"Reading {labels_csv}...")
    with open(labels_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    data_rows = reader[1:]
    
    # Track missing trees
    missing_trees_jsons = []
    
    logging.info("Step 1: Moving extracted text JSON files to final directory...")
    moved_jsons = 0
    already_moved_jsons = 0
    
    for row in data_rows:
        cik = str(row[0]).zfill(10)
        year = str(row[3])
        
        json_filename = f"{cik}_{year}_10-K_extracted.json"
        src_json = source_extracted_dir / json_filename
        dest_json = final_extracted_dir / json_filename
        
        # 1. Move the extracted text
        if src_json.exists():
            shutil.move(str(src_json), str(dest_json))
            moved_jsons += 1
        elif dest_json.exists():
            already_moved_jsons += 1
        else:
            # Fallback to check unpadded CIK if needed
            alt_json_filename = f"{int(cik)}_{year}_10-K_extracted.json"
            alt_src_json = source_extracted_dir / alt_json_filename
            if alt_src_json.exists():
                shutil.move(str(alt_src_json), str(dest_json)) # move to the padded standardized name
                moved_jsons += 1
            else:
                logging.warning(f"Could not find extracted JSON for {cik} {year}")
                continue
                
        # 2. Check if tree exists in inference_trees_directory
        tree_folder_name = dest_json.stem  # e.g., 0001786248_2022_10-K_extracted
        tree_dir = inference_trees_dir / tree_folder_name
        tree_file = tree_dir / "tree.pkl"
        
        if not tree_file.exists():
            # We need to build this tree
            missing_trees_jsons.append(dest_json)
                
    logging.info(f"Moved {moved_jsons} new JSON files. ({already_moved_jsons} already present).")
    
    # Step 2: Build missing trees
    if len(missing_trees_jsons) == 0:
        logging.info("All unrated companies have their RAPTOR trees successfully built! Nothing else to do.")
        return
        
    logging.info(f"Found {len(missing_trees_jsons)} unrated companies missing their RAPTOR trees.")
    logging.info("Initializing RAPTOR Summarizer (ensure your llama.cpp server is running on port 8001!)...")
    
    # Lazy import so we don't load huge models if not needed
    from src.raptor_pipeline import RaptorPipelineConfig, RaptorSummarizer
    from src.raptor_pipeline.build_tree import save_tree
    
    config = RaptorPipelineConfig(
        max_tokens=512,
        num_layers=5,
        summarizer_model_name="meta-llama/Llama-3.1-8B-Instruct",
        embedding_model_name="BAAI/bge-base-en-v1.5",
    )
    summarizer = RaptorSummarizer(config)
    
    for idx, json_path in enumerate(missing_trees_jsons, 1):
        logging.info(f"Building RAPTOR tree for {json_path.name} ({idx}/{len(missing_trees_jsons)})")
        try:
            tree = summarizer.build_tree_from_json(json_path)
            # Save the tree inside inference_trees_directory / {stem}
            save_tree(tree, inference_trees_dir / json_path.stem)
        except Exception as e:
            logging.error(f"Failed to build tree for {json_path.name}: {str(e)}")

    logging.info("Finished building inference trees!")

if __name__ == '__main__':
    main()
