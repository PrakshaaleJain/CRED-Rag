import os
import json
import logging
import requests
from pathlib import Path
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.docstore.document import Document

from src.raptor_pipeline.tree_builder import ClusterTreeBuilder, TreeBuilderConfig
from src.raptor_pipeline.utils import get_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

VLLM_API_URL = "http://localhost:8001/v1/chat/completions"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def generate_answer(query: str, contexts: list[str]) -> str:
    context_str = "\n\n".join(contexts)
    prompt = (
        f"Answer the following question based ONLY on the provided context.\n"
        f"If the answer is not in the context, reply 'I cannot answer based on the context.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise financial assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    
    try:
        response = requests.post(VLLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"vLLM API error: {e}")
        return "ERROR: vLLM Server Offline or Timeout"

def build_naive_faiss(text: str, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

def build_raptor_faiss(text: str, embeddings):
    config = TreeBuilderConfig(
        max_tokens=512,
        num_layers=3,
        summarization_length=256
    )
    builder = ClusterTreeBuilder(config)
    try:
        tree = builder.build_from_text(text)
        all_texts = [node.text for node in tree.all_nodes.values()]
    except Exception as e:
        logging.error(f"RAPTOR Tree generation failed (likely due to offline vLLM): {e}")
        all_texts = ["Error: Could not generate RAPTOR summaries due to offline vLLM server."]
        
    return FAISS.from_texts(all_texts, embeddings)

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    
    benchmark_path = data_dir / 'raptor_eval_benchmark.json'
    results_path = data_dir / 'ab_benchmark_results.json'
    
    if not benchmark_path.exists():
        logging.error("Benchmark JSON not found. Run build_eval_dataset.py first.")
        return
        
    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)
        
    logging.info("Loading BGE Embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load texts
    texts = {}
    for company in ["HAVA", "ZETA"]:
        md_file = data_dir / 'sec_filings_md' / f"{company}_2026_10K.md"
        if md_file.exists():
            with open(md_file, 'r', encoding='utf-8') as f:
                texts[company] = f.read()
        else:
            logging.warning(f"Could not find {md_file}")
            texts[company] = ""
            
    # Build Indices
    indices = {}
    for company, text in texts.items():
        if not text:
            continue
        logging.info(f"Building Naive FAISS for {company}...")
        naive_idx = build_naive_faiss(text, embeddings)
        logging.info(f"Building RAPTOR FAISS for {company}...")
        raptor_idx = build_raptor_faiss(text, embeddings)
        indices[company] = {"naive": naive_idx, "raptor": raptor_idx}
        
    results = []
    
    logging.info("Starting A/B Execution Loop...")
    for item in tqdm(benchmark, desc="Evaluating Queries"):
        company = item["company_id"]
        query = item["query"]
        
        if company not in indices:
            logging.warning(f"Skipping {company} due to missing data.")
            continue
            
        # Naive RAG
        naive_idx = indices[company]["naive"]
        naive_docs = naive_idx.similarity_search(query, k=5)
        naive_contexts = [doc.page_content for doc in naive_docs]
        naive_answer = generate_answer(query, naive_contexts)
        
        # RAPTOR RAG
        raptor_idx = indices[company]["raptor"]
        raptor_docs = raptor_idx.similarity_search(query, k=5)
        raptor_contexts = [doc.page_content for doc in raptor_docs]
        raptor_answer = generate_answer(query, raptor_contexts)
        
        results.append({
            "question_id": item["id"],
            "company_id": company,
            "abstraction_level": item["abstraction_level"],
            "query": query,
            "ground_truth": item["ground_truth"],
            "expected_keywords": item["expected_keywords"],
            "naive_retrieved_contexts": naive_contexts,
            "naive_generated_answer": naive_answer,
            "raptor_retrieved_contexts": raptor_contexts,
            "raptor_generated_answer": raptor_answer
        })
        
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"A/B Benchmark complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
