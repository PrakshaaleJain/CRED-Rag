import os
import json
import csv
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Credit Rating Mapping (21-point scale)
RATING_MAP = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1, 'D': 0
}
REVERSE_RATING_MAP = {v: k for k, v in RATING_MAP.items()}

def load_features_text(json_path: Path) -> str:
    """Concatenate qualitative JSON features into a single string for embedding."""
    if not json_path.exists():
        return ""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return " ".join(str(v) for v in data.values())
    except Exception as e:
        logging.error(f"Failed to load {json_path}: {e}")
        return ""

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    
    rated_csv = project_root / 'data' / 'final_training_labels.csv'
    unrated_csv = project_root / 'data' / 'unrated_inference_set.csv'
    
    rated_features_dir = project_root / 'data' / 'qualitative_features'
    unrated_features_dir = project_root / 'data' / 'inference_features'
    
    out_csv = project_root / 'data' / 'knn_proxy_predictions.csv'
    
    # 1. Load the embedding model
    logging.info("Loading BAAI/bge-base-en-v1.5 embedding model...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # 2. Build Knowledge Base (Rated Companies)
    logging.info("Building knowledge base of rated companies...")
    kb_ciks, kb_years, kb_ratings, kb_texts = [], [], [], []
    
    with open(rated_csv, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))[1:]
        for row in reader:
            cik, year, rating = str(row[0]).zfill(10), str(row[3]), str(row[4])
            source = str(row[6]) if len(row) > 6 else '1'
            if rating == 'NR' or source == '0':
                continue
                
            features_file = rated_features_dir / f"{cik}_{year}_features.json"
            text = load_features_text(features_file)
            if text:
                kb_ciks.append(cik)
                kb_years.append(year)
                kb_ratings.append(RATING_MAP.get(rating, 10)) # default middle if unknown
                kb_texts.append(text)
                
    logging.info(f"Embedding {len(kb_texts)} rated companies...")
    kb_embeddings = embedder.encode(kb_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # 3. Process Inference Set (Unrated Companies)
    logging.info("Processing unrated companies...")
    inf_ciks, inf_years, inf_texts = [], [], []
    
    with open(unrated_csv, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))[1:]
        for row in reader:
            cik, year = str(row[0]).zfill(10), str(row[3])
            features_file = unrated_features_dir / f"{cik}_{year}_features.json"
            text = load_features_text(features_file)
            if text:
                inf_ciks.append(cik)
                inf_years.append(year)
                inf_texts.append(text)
                
    if not inf_texts:
        logging.error("No unrated inference features found. Please run extract_inference_features.py first.")
        return
        
    logging.info(f"Embedding {len(inf_texts)} unrated companies...")
    inf_embeddings = embedder.encode(inf_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # 4. Perform KNN Cosine Similarity
    logging.info("Calculating K-Nearest Neighbors (K=5)...")
    K = 5
    similarities = cosine_similarity(inf_embeddings, kb_embeddings)
    
    results = []
    
    for i in range(len(inf_texts)):
        sim_scores = similarities[i]
        top_k_idx = np.argsort(sim_scores)[-K:][::-1] # descending
        
        peer_ratings = [kb_ratings[idx] for idx in top_k_idx]
        avg_rating_score = round(float(np.mean(peer_ratings)))
        predicted_rating = REVERSE_RATING_MAP.get(avg_rating_score, 'NR')
        
        row_data = [
            inf_ciks[i], inf_years[i], predicted_rating, avg_rating_score
        ]
        
        # Add peer details
        for idx in top_k_idx:
            row_data.extend([kb_ciks[idx], REVERSE_RATING_MAP.get(kb_ratings[idx]), round(float(sim_scores[idx]), 4)])
            
        results.append(row_data)
        
    # 5. Export
    logging.info(f"Exporting predictions to {out_csv}...")
    headers = [
        "CIK", "Year", "Predicted_Rating", "Avg_Rating_Score",
        "Peer1_CIK", "Peer1_Rating", "Peer1_Sim",
        "Peer2_CIK", "Peer2_Rating", "Peer2_Sim",
        "Peer3_CIK", "Peer3_Rating", "Peer3_Sim",
        "Peer4_CIK", "Peer4_Rating", "Peer4_Sim",
        "Peer5_CIK", "Peer5_Rating", "Peer5_Sim"
    ]
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
        
    logging.info("Done!")

if __name__ == '__main__':
    main()
