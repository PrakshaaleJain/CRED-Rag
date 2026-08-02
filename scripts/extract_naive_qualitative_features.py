#!/usr/bin/env python3
import os
import csv
import json
import logging
import requests
from pathlib import Path
import sys
import itertools

# Ensure we can import from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.raptor_pipeline.load_tree import load_tree
from sentence_transformers import SentenceTransformer, util
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Topics and their embedding queries
TOPICS = {
    "Revenue": "Revenue: What is driving sales growth or contraction? (Look for high-level market trends and granular product/segment details).",
    "Operating Profit": "Operating Profit: What factors are impacting operational efficiency and core business profitability?",
    "Net/Gross Margins": "Net/Gross Margins: How is pricing power, inflation, or cost of goods sold (COGS) affecting their margins?",
    "Net Profit": "Net Profit: What is management's narrative around bottom-line earnings, taxes, and one-time expenses?",
    "Free Cash Flow": "Free Cash Flow: What is the commentary regarding cash generation, capital expenditures, debt repayment, and liquidity?"
}

LLM_API_URL = "http://localhost:8001/v1/chat/completions"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

def construct_prompt(dimension: str, context: str) -> str:
    return f"""System: You are an expert corporate credit risk analyst conducting a rigorous financial audit. I will provide you with a comprehensive summary of a company's SEC 10-K filing. Your task is to perform targeted qualitative extraction for exactly one specific financial dimension: {dimension}.

Carefully analyze the text and extract any management commentary, forward-looking guidance, or identified risk factors that directly impact this specific dimension. Your extraction must be entirely grounded in the provided text. Do not summarize other metrics or hallucinate external macroeconomic factors. 

If the provided summary contains no relevant information regarding {dimension}, you must output exactly the word "None".

User: {context}
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_company(row, out_dir, trees_dir, model_id, topic_embeddings, embedder, api_url):
    cik = str(row[0]).zfill(10)
    year = str(row[3])
    rating = str(row[4])
    source_indicator = str(row[6]) if len(row) > 6 else '1'
    
    if rating == 'NR' or source_indicator == '0':
        return None
        
    out_file = out_dir / f"{cik}_{year}_features.json"
    if out_file.exists():
        # Already processed, silent return to prevent log spam when running concurrently
        return f"{cik}_{year} (already processed)"
        
    tree_folder = trees_dir / f"{cik}_{year}_10-K_extracted"
    if not tree_folder.exists():
        return f"Tree not found for {cik}_{year}"
        
    try:
        tree = load_tree(tree_folder)
    except Exception as e:
        return f"Failed to load tree for {cik}_{year}: {e}"
        
    node_ids = []
    node_embs = []
    
    # NAIVE RAG: Only use leaf_nodes (raw text chunks), ignore summaries
    for n_id, node in tree.leaf_nodes.items():
        if node.embeddings:
            emb_key = list(node.embeddings.keys())[0]
            node_ids.append(n_id)
            node_embs.append(node.embeddings[emb_key])
            
    if not node_embs:
        return f"No valid embeddings found in tree for {cik}_{year}."
        
    device = list(topic_embeddings.values())[0].device
    corpus_tensor = torch.tensor(node_embs, device=device)
    base_temperature = 0.1
    for attempt in range(3):
        current_k = 2 if attempt == 0 else 1
        
        selected_nodes = set()
        for topic_name, q_emb in topic_embeddings.items():
            cos_scores = util.cos_sim(q_emb, corpus_tensor)[0]
            top_results = torch.topk(cos_scores, k=min(current_k, len(node_ids)))
            
            for score, idx_tensor in zip(top_results[0], top_results[1]):
                selected_chunks.add(node_ids[idx_tensor.item()])
                
        # Retrieve texts from leaf_nodes
        retrieved_texts = [chunk_mapping[c_id] for c_id in selected_chunks if c_id in chunk_mapping]
        context_str = "\n\n---\n\n".join(retrieved_texts)
        
        parsed_json = {}
        dimensions = ["Revenue", "Operating Profit", "Net/Gross Margins", "Net Profit", "Free Cash Flow"]
        
        try:
            for dim in dimensions:
                prompt = construct_prompt(dim, context_str)
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": base_temperature + (attempt * 0.2),
                    "max_tokens": 2048
                }
                resp = requests.post(api_url, json=payload, timeout=600)
                resp.raise_for_status()
                parsed_json[dim] = resp.json()["choices"][0]["message"]["content"].strip()
            
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            
            return f"Successfully processed {cik}_{year}"
        except Exception as e:
            if attempt == 2:
                return f"LLM API failed for {cik}_{year} after 3 retries: {e}"
            logging.warning(f"LLM extraction failed for {cik}_{year}, retrying... (attempt {attempt + 1})")

def main():
    out_dir = project_root / 'data' / 'naive_qualitative_features'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    trees_dir = project_root / 'data' / 'final_trees_directory'
    
    if not labels_csv.exists():
        logging.error(f"{labels_csv} not found.")
        return
        
    try:
        base_url = LLM_API_URL.replace("/chat/completions", "/models")
        res = requests.get(base_url, timeout=3)
        res.raise_for_status()
        model_id = res.json()["data"][0]["id"]
        logging.info(f"Connected to LLM Server. Using model: {model_id}")
    except Exception as e:
        logging.error(f"Cannot connect to local LLM server at {LLM_API_URL}: {e}")
        logging.error("Please ensure llama-cpp-python server is running.")
        return

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    topic_embeddings = {k: embedder.encode(v, convert_to_tensor=True) for k, v in TOPICS.items()}
    
    logging.info("Reading labels...")
    with open(labels_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    data_rows = reader[1:]
    
    logging.info(f"Beginning concurrent Naive RAG processing of {len(data_rows)} companies...")
    
    # Process with 6 concurrent threads to saturate the single LLM server's batch queue
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                process_company, row, out_dir, trees_dir, model_id, topic_embeddings, embedder, LLM_API_URL
            ): row for row in data_rows
        }
        
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                logging.info(f"[{idx}/{len(data_rows)}] {result}")
            
    logging.info("Naive Batch extraction complete!")
if __name__ == '__main__':
    main()
