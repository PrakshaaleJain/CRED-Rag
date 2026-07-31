import os
import json
import torch
import logging
import pandas as pd
from pathlib import Path
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    project_root = Path(__file__).resolve().parents[1]
    features_dir = project_root / 'data' / 'qualitative_features'
    
    csv_out = project_root / 'data' / 'qualitative_sentiment_scores.csv'
    json_out = project_root / 'data' / 'qualitative_sentiment_scores.json'
    
    if not features_dir.exists():
        logging.error(f"Directory {features_dir} does not exist!")
        return

    # 1. Initialize Pipeline ONCE with FP16 and GPU batching
    logging.info("Initializing ProsusAI/finbert pipeline...")
    device = 0 if torch.cuda.is_available() else -1
    
    # Use return_all_scores=True or top_k=None depending on transformers version
    try:
        finbert = pipeline(
            "text-classification", 
            model="ProsusAI/finbert", 
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            return_all_scores=True
        )
    except TypeError:
        # Fallback for newer transformers versions
        finbert = pipeline(
            "text-classification", 
            model="ProsusAI/finbert", 
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            top_k=None
        )

    # 2. Collect all text across all companies
    logging.info("Collecting qualitative text from JSON files...")
    
    topics = [
        "Revenue", 
        "Operating Profit", 
        "Net/Gross Margins", 
        "Net Profit", 
        "Free Cash Flow"
    ]
    
    # Store pre-calculated scores (e.g. for "No commentary found")
    final_scores = []
    
    # Data structure to hold items needing inference
    inference_texts = []
    inference_metadata = [] # stores (cik, year, topic)
    
    for json_file in features_dir.glob("*_features.json"):
        # File name format: {cik}_{year}_features.json
        parts = json_file.stem.split('_')
        cik = parts[0]
        year = parts[1]
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {json_file}: {e}")
            continue
            
        row_dict = {"Company_ID": cik, "Year": year}
        
        for topic in topics:
            val = data.get(topic, "")
            
            # Concatenate list of sentences if needed
            if isinstance(val, list):
                text_content = " ".join(str(s) for s in val)
            else:
                text_content = str(val)
                
            if "No qualitative commentary found" in text_content or not text_content.strip():
                # Bypass inference, assign 0.0 directly
                row_dict[f"{topic}_Sentiment"] = 0.0
            else:
                # Needs inference
                inference_texts.append(text_content)
                inference_metadata.append((cik, year, topic))
                
        final_scores.append(row_dict)
        
    logging.info(f"Found {len(inference_texts)} text blocks requiring FinBERT inference.")
    
    # 3. Batch Inference on GPU
    logging.info("Running batched inference (batch_size=128, max_length=512)...")
    
    # We create a dictionary to easily update the final_scores rows later
    row_lookup = {f"{row['Company_ID']}_{row['Year']}": row for row in final_scores}
    
    if inference_texts:
        # Run the pipeline iterator
        results = finbert(inference_texts, batch_size=128, truncation=True, max_length=512)
        
        # Calculate continuous scores
        for idx, result in enumerate(results):
            cik, year, topic = inference_metadata[idx]
            
            # The result is a list of dictionaries [{'label': 'positive', 'score': 0.8}, ...]
            prob_pos = 0.0
            prob_neg = 0.0
            
            for score_dict in result:
                label = score_dict['label'].lower()
                if label == 'positive':
                    prob_pos = score_dict['score']
                elif label == 'negative':
                    prob_neg = score_dict['score']
                    
            continuous_score = prob_pos - prob_neg
            continuous_score = round(continuous_score, 4)
            
            # Update the appropriate row dict
            row_lookup[f"{cik}_{year}"][f"{topic}_Sentiment"] = continuous_score

    # 4. Aggregation and Output
    logging.info("Formatting output...")
    df = pd.DataFrame(final_scores)
    
    # Ensure correct column order
    cols = ["Company_ID", "Year"] + [f"{t}_Sentiment" for t in topics]
    # Handle missing columns safely
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[cols]
    
    # Save CSV
    df.to_csv(csv_out, index=False)
    
    # Save JSON (hierarchical)
    hierarchical_dict = {}
    for _, row in df.iterrows():
        cik = str(row['Company_ID'])
        year = str(row['Year'])
        if cik not in hierarchical_dict:
            hierarchical_dict[cik] = {}
            
        hierarchical_dict[cik][year] = {
            "Revenue": row["Revenue_Sentiment"],
            "Operating Profit": row["Operating Profit_Sentiment"],
            "Net/Gross Margins": row["Net/Gross Margins_Sentiment"],
            "Net Profit": row["Net Profit_Sentiment"],
            "Free Cash Flow": row["Free Cash Flow_Sentiment"]
        }
        
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(hierarchical_dict, f, indent=4)
        
    logging.info(f"Success! Saved outputs to:\n - {csv_out}\n - {json_out}")
    print("\nSample Output (First 5 Rows):")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
