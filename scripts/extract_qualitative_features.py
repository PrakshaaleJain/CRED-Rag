#!/usr/bin/env python3
import os
import csv
import json
import logging
import requests
from pathlib import Path
import sys

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

def construct_prompt(context: str) -> str:
    return f"""You are an expert corporate credit risk analyst. I will provide you with a multi-resolution context retrieved from an SEC 10-K filing. This context contains a mix of high-level abstract summaries and highly specific granular text chunks. Your task is to synthesize this information to extract the company management's qualitative commentary and forward-looking statements regarding specific financial health indicators.

Your task is to synthesize this information and output a JSON object with EXACTLY 5 keys:
"Revenue", "Operating Profit", "Net/Gross Margins", "Net Profit", "Free Cash Flow".
The value for each key should be your qualitative summary. Do not output anything outside of the JSON object.

Based on the provided multi-resolution 10-K context, extract and summarize management's commentary, explanations, and strategic outlook on the following five topics:

Revenue: What is driving sales growth or contraction? (Look for high-level market trends and granular product/segment details).
Operating Profit: What factors are impacting operational efficiency and core business profitability?
Net/Gross Margins: How is pricing power, inflation, or cost of goods sold (COGS) affecting their margins?
Net Profit: What is management's narrative around bottom-line earnings, taxes, and one-time expenses?
Free Cash Flow: What is the commentary regarding cash generation, capital expenditures, debt repayment, and liquidity?

Constraints:
- Synthesize the information intelligently. Use the high-level summaries for the overarching narrative, and use the granular chunks to provide specific examples.
- Do not just list the numbers. I need the qualitative context—the 'why' behind the numbers.
- If a topic is not discussed in the text, explicitly state 'No qualitative commentary found for this topic.'

Context:
{context}
"""

def main():
    out_dir = project_root / 'data' / 'qualitative_features'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels_csv = project_root / 'data' / 'final_training_labels.csv'
    trees_dir = project_root / 'data' / 'final_trees_directory'
    
    if not labels_csv.exists():
        logging.error(f"{labels_csv} not found.")
        return
        
    # Check if the LLM server is up
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
    
    for idx, row in enumerate(data_rows, 1):
        cik = str(row[0]).zfill(10)
        year = str(row[3])
        rating = str(row[4])
        source_indicator = str(row[6]) if len(row) > 6 else '1'
        
        if rating == 'NR' or source_indicator == '0':
            continue
            
        out_file = out_dir / f"{cik}_{year}_features.json"
        if out_file.exists():
            logging.info(f"[{idx}/{len(data_rows)}] Skipping {cik}_{year} (already processed)")
            continue
            
        tree_folder = trees_dir / f"{cik}_{year}_10-K_extracted"
        if not tree_folder.exists():
            logging.warning(f"[{idx}/{len(data_rows)}] Tree not found for {cik}_{year}. Skipping.")
            continue
            
        logging.info(f"[{idx}/{len(data_rows)}] Processing {cik}_{year}...")
        try:
            tree = load_tree(tree_folder)
        except Exception as e:
            logging.error(f"Failed to load tree for {cik}_{year}: {e}")
            continue
            
        # Gather all nodes and their embeddings
        node_ids = []
        node_embs = []
        
        for n_id, node in tree.all_nodes.items():
            # The custom RAPTOR pipeline hardcodes the embedding key as 'FinE5' by default in TreeBuilderConfig
            # even when using BAAI/bge-base-en-v1.5. We will grab whatever embedding key is present.
            if node.embeddings:
                emb_key = list(node.embeddings.keys())[0]
                node_ids.append(n_id)
                node_embs.append(node.embeddings[emb_key])
                
        if not node_embs:
            logging.warning(f"No valid embeddings found in tree for {cik}_{year}.")
            continue
            
        # Ensure corpus_tensor is on the same device as the query embeddings (e.g., cuda:0)
        device = list(topic_embeddings.values())[0].device
        corpus_tensor = torch.tensor(node_embs, device=device)
        selected_nodes = set()
        
        # Pull top 4 nodes per topic
        for topic_name, q_emb in topic_embeddings.items():
            cos_scores = util.cos_sim(q_emb, corpus_tensor)[0]
            top_results = torch.topk(cos_scores, k=min(4, len(node_ids)))
            
            for score, idx_tensor in zip(top_results[0], top_results[1]):
                selected_nodes.add(node_ids[idx_tensor.item()])
                
        # Combine retrieved text
        retrieved_texts = [tree.all_nodes[n].text for n in selected_nodes]
        context_str = "\n\n---\n\n".join(retrieved_texts)
        
        prompt = construct_prompt(context_str)
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        
        try:
            # Increased timeout to 600 seconds to prevent Read timed out on dense documents
            resp = requests.post(LLM_API_URL, json=payload, timeout=600)
            resp.raise_for_status()
            llm_output = resp.json()["choices"][0]["message"]["content"]
            
            # Verify it's valid JSON
            parsed_json = json.loads(llm_output)
            
            # Save it
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"LLM API failed for {cik}_{year}: {e}")
            
    logging.info("Batch extraction complete!")

if __name__ == '__main__':
    main()
