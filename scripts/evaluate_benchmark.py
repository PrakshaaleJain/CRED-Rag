import os
import json
import csv
import logging
import requests
import re
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

LLAMA_API_URL = "http://localhost:8001/v1/chat/completions"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

EVAL_PROMPT_TEMPLATE = """You are an expert RAG evaluation judge. You will be provided with a Question, the expected Ground Truth, the Retrieved Contexts, and the Generated Answer from a RAG pipeline.

Evaluate the RAG pipeline's performance on a scale of 1 to 5 for the following four metrics:
1. Context Precision: How relevant is the retrieved context to the query? (1 = completely irrelevant, 5 = highly relevant and precise)
2. Context Recall: Does the retrieved context contain all the facts needed to form the ground truth? (1 = completely missing facts, 5 = contains all facts)
3. Faithfulness: Is the generated answer derived strictly from the retrieved context without making up facts? (1 = total hallucination, 5 = perfectly faithful to context)
4. Answer Relevance: Does the generated answer directly and accurately address the user query compared to the ground truth? (1 = completely irrelevant or wrong, 5 = perfect answer)

Input:
- Question: {query}
- Ground Truth: {ground_truth}
- Retrieved Contexts: {context}
- Generated Answer: {answer}

Output ONLY a valid JSON object with the exact keys: 'context_precision', 'context_recall', 'faithfulness', 'answer_relevance'. Values must be integers between 1 and 5. Do not include any other text.
"""

def call_llm_judge(query, ground_truth, contexts, answer) -> dict:
    context_str = "\n---\n".join(contexts[:3]) # Limit to top 3 to save tokens for judging
    prompt = EVAL_PROMPT_TEMPLATE.format(
        query=query,
        ground_truth=ground_truth,
        context=context_str,
        answer=answer
    )
    
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a JSON-only evaluation bot. Output ONLY raw valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    
    default_scores = {
        "context_precision": 0,
        "context_recall": 0,
        "faithfulness": 0,
        "answer_relevance": 0
    }
    
    try:
        response = requests.post(LLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        # Find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return default_scores
    except Exception as e:
        # logging.error(f"vLLM Judge error: {e}")
        return default_scores

def calculate_metrics(results_data):
    evaluated_results = []
    
    logging.info("Starting LLM-as-a-Judge Evaluation...")
    for item in tqdm(results_data, desc="Scoring Results"):
        query = item["query"]
        ground_truth = item["ground_truth"]
        
        # Eval Naive
        naive_scores = call_llm_judge(
            query, ground_truth, 
            item["naive_retrieved_contexts"], 
            item["naive_generated_answer"]
        )
        
        # Eval RAPTOR
        raptor_scores = call_llm_judge(
            query, ground_truth, 
            item["raptor_retrieved_contexts"], 
            item["raptor_generated_answer"]
        )
        
        evaluated_results.append({
            "question_id": item["question_id"],
            "company_id": item["company_id"],
            "abstraction_level": item["abstraction_level"],
            "naive_precision": naive_scores.get("context_precision", 0),
            "naive_recall": naive_scores.get("context_recall", 0),
            "naive_faithfulness": naive_scores.get("faithfulness", 0),
            "naive_relevance": naive_scores.get("answer_relevance", 0),
            "raptor_precision": raptor_scores.get("context_precision", 0),
            "raptor_recall": raptor_scores.get("context_recall", 0),
            "raptor_faithfulness": raptor_scores.get("faithfulness", 0),
            "raptor_relevance": raptor_scores.get("answer_relevance", 0)
        })
        
    return evaluated_results

def print_summary_table(evaluated_results):
    grouped = {}
    for res in evaluated_results:
        lvl = res["abstraction_level"]
        if lvl not in grouped:
            grouped[lvl] = []
        grouped[lvl].append(res)
        
    print("\n### RAG Triad Evaluation Summary\n")
    print("| Abstraction Level | Pipeline | Context Precision | Context Recall | Faithfulness | Answer Relevance |")
    print("|-------------------|----------|-------------------|----------------|--------------|------------------|")
    
    for lvl in ["Factual", "Conceptual", "Strategic"]:
        items = grouped.get(lvl, [])
        if not items:
            continue
            
        n = len(items)
        n_p = sum([x["naive_precision"] for x in items]) / n
        n_r = sum([x["naive_recall"] for x in items]) / n
        n_f = sum([x["naive_faithfulness"] for x in items]) / n
        n_a = sum([x["naive_relevance"] for x in items]) / n
        
        r_p = sum([x["raptor_precision"] for x in items]) / n
        r_r = sum([x["raptor_recall"] for x in items]) / n
        r_f = sum([x["raptor_faithfulness"] for x in items]) / n
        r_a = sum([x["raptor_relevance"] for x in items]) / n
        
        print(f"| **{lvl}** | Naive RAG | {n_p:.4f} | {n_r:.4f} | {n_f:.4f} | {n_a:.4f} |")
        print(f"| | **RAPTOR** | **{r_p:.4f}** | **{r_r:.4f}** | **{r_f:.4f}** | **{r_a:.4f}** |")

def save_csv_and_errors(evaluated_results, out_dir):
    csv_path = out_dir / 'rag_evaluation_scores.csv'
    error_log_path = out_dir / 'retrieval_failure_cases.txt'
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = list(evaluated_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluated_results:
            writer.writerow(row)
            
    # Save error log where RAPTOR underperformed Naive on Answer Relevance
    with open(error_log_path, 'w') as f:
        for row in evaluated_results:
            if row["raptor_relevance"] < row["naive_relevance"]:
                f.write(f"Question ID: {row['question_id']} ({row['abstraction_level']})\n")
                f.write(f"Naive Relevance: {row['naive_relevance']} | RAPTOR Relevance: {row['raptor_relevance']}\n")
                f.write("---\n")
                
    logging.info(f"Saved CSV to {csv_path}")
    logging.info(f"Saved Error Log to {error_log_path}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    out_dir = project_root / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = data_dir / 'ab_benchmark_results.json'
    
    if not results_path.exists():
        logging.error("ab_benchmark_results.json not found. Run run_ab_benchmark.py first.")
        return
        
    with open(results_path, 'r') as f:
        results_data = json.load(f)
        
    evaluated_results = calculate_metrics(results_data)
    
    if evaluated_results:
        print_summary_table(evaluated_results)
        save_csv_and_errors(evaluated_results, out_dir)
        
if __name__ == "__main__":
    main()
