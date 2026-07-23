# Academic Comparative Analysis of Financial Text Models & Recommendations

This report compares **summarization** and **embedding** models for processing financial texts (e.g., SEC 10-K filings, earnings transcripts, news), provides recommendations for the **CRED-RAPTOR** pipeline, and lists formal citations from top-tier AI and NLP conferences.

---

## 1. Recommendations for CRED-RAPTOR

To optimize the hierarchical tree building and subsequent RAG retrieval in CRED-RAPTOR, we recommend three distinct configurations based on your operational trade-offs (data privacy/cost vs. performance):

### Option A: Best Open-Source Local Setup (Recommended)
This configuration offers maximum data privacy, zero API costs, and runs locally.
* **Summarization Model:** **`Qwen2.5-7B-Instruct`** (or **`Llama-3.1-8B-Instruct`**)
  * *Why:* High ROUGE precision and semantic BERTScore on financial report metrics (Yang et al., 2024), without the extreme latency and VRAM footprint of larger models.
* **Embedding Model:** **`Fin-E5`** (or **`gte-Qwen2-1.5B-instruct`** for lower hardware specs)
  * *Why:* `Fin-E5` (based on E5-Mistral-7B) ranks at the top of the FinMTEB benchmark (Tang & Yang, 2025). It uses persona-based instruction tuning specifically to understand financial reports, providing superior semantic tree grouping compared to standard `BGE` or `MiniLM`.

### Option B: Best Proprietary Cloud Setup (Maximum Performance)
This configuration offers the highest retrieval and reasoning accuracy but incurs API costs and sends data to external providers.
* **Summarization Model:** **`Claude 3.5 Sonnet`**
  * *Why:* Unmatched reasoning capabilities, complex numerical compliance, and structured extraction when writing abstracts for SEC filings.
* **Embedding Model:** **`Voyage-Finance-2`** (Voyage AI)
  * *Why:* Tailored specifically for financial retrieval and RAG pipelines. It supports a **32K token context window** (allowing ingestion of entire SEC sections in one go) and outperforms general models (like OpenAI `text-embedding-3`) on FinanceBench.

### Option C: Specialized Financial Fine-Tuning Setup
* **Summarization Model:** **`FinGPT-v3`** (or other FinGPT variants based on LLaMA/Qwen)
  * *Why:* Directly fine-tuned on financial narratives and sentiment analysis (Yang et al., 2023).
* **Embedding Model:** **`Fin-E5`** (Open-source, local).

---

## 2. Extractive vs. Abstractive Summarization in RAPTOR

Although RAPTOR stands for *Recursive Abstractive Processing for Tree-Organized Retrieval*, **extractive methods can absolutely be used** as the summarization engine for tree construction. 

### How Extractive RAPTOR Works
In a standard RAPTOR build phase, leaf node chunks are clustered together, and an LLM is prompted to write an abstractive summary of that cluster. If replaced with an extractive system:
1. **Cluster Merging:** Chunks grouped into a cluster (e.g., 5 chunks of 200 words each = 1,000 words total) are merged.
2. **Sentence Ranking:** An extractive algorithm ranks individual sentences within the cluster based on relevance or centrality.
3. **Sentence Selection:** The top $N$ sentences (up to a token limit, e.g., 150 tokens) are selected and concatenated verbatim to form the parent node's text.
4. **Hierarchical Recursion:** The parent node is embedded and clustered again at the next layer of the tree.

### Comparative Trade-offs for Credit Risk Assessment

| Metric / Dimension | Abstractive RAPTOR (LLM-based) | Extractive RAPTOR (Algorithmic) |
| :--- | :--- | :--- |
| **Factual Accuracy** | **Risky.** LLMs can hallucinate numbers, swap metrics (e.g., EBIT vs. EBITDA), or misattribute risks. | **Perfect (100%).** Copies sentences verbatim from the source; zero hallucination risk. |
| **Synthesis Quality** | **Excellent.** Combines disjoint facts into a single, flowing, readable theme. | **Poor.** Can result in a disjointed list of sentences that lack transition context. |
| **Computational Cost** | **High.** Requires hundreds of LLM API calls or GPU inferences during tree indexing. | **Negligible.** Algorithms (TextRank, LexRank) run locally on CPU in milliseconds. |
| **Traceability** | **Indirect.** Requires semantic similarity mapping to trace parent claims back to source text. | **Direct.** Simple substring matching links parent sentences directly to leaf chunks. |
| **Information Density** | **High.** Condenses and abstracts large paragraphs into a few key points. | **Low.** Retains full sentence structures, which increases token footprint. |

---

## 3. Which Summarization is Best for Qualitative Signal Scoring?

For the specific task of retrieving summaries to **score qualitative signals** (e.g., Business Model Quality, Industry Risks, Management Guidance Stability) using a scoring LLM, **Abstractive Summarization makes significantly more sense** than extractive summarization. 

Here is why:

### 1. Thematic Synthesis vs. Disjointed Details
* **Abstractive:** Qualitative analysis requires understanding thematic trends, overall sentiment, and management tone. Abstractive summarization excels at distilling multi-page arguments into unified, high-level narratives. It bridges scattered signals (e.g., a sentence on supply chain issues in Section A and a sentence on raw material costs in Section B) into a consolidated risk description.
* **Extractive:** Extractive models rank sentences in isolation. If a qualitative signal is spread across several paragraphs, an extractive summarizer will extract a few disconnected sentences, losing the context that links them. This forces the downstream scoring LLM to spend extra tokens synthesizing disjointed sentences.

### 2. High Information Density & Context Budgets
* **Abstractive:** Condenses qualitative concepts into highly dense bullet points or paragraphs. This allows you to retrieve more thematic coverage within the scoring LLM's context window.
* **Extractive:** Retains full sentence structures, repetitions, and boilerplate legal phrasing. This consumes a massive token footprint, limiting how much historical or multi-section context you can feed to the scoring LLM at once.

### 3. Reduced Hallucination Risk for Qualitative Data
* The primary danger of abstractive models is the hallucination of **quantitative figures** (e.g., misreporting capital expenditure numbers). 
* **Qualitative concepts** (e.g., "The company faces supply chain execution risks in Europe due to energy costs") are represented as semantic themes. LLMs are highly robust at summarizing qualitative ideas without shifting their meaning, making the "hallucination penalty" much lower for qualitative scoring than it is for mathematical validation.

### Recommended Hybrid Strategy for Scoring
If you want to ensure the qualitative scoring is both rich in semantic context and factually anchored, use an **abstractive summarization engine** during RAPTOR tree construction, but prompt the downstream **scoring LLM** to cite exact quotes or verify key sentences from the raw (leaf-level) chunks before finalizing the score.

---

## 4. Web Comparisons of Financial Abstractive Summarization Models

Several peer-reviewed papers compare abstractive models on corporate financial documents. Here are the core comparative results from key benchmarks:

### Study 1: EMNLP Findings (FINDSum Dataset)
* **Objective:** Evaluating sequence-to-sequence abstractive models on multi-table and long-text financial annual reports (21,125 reports).
* **Models Compared:** **BART-base**, **BART-large**, **PEGASUS-base**, and **LED** (Longformer-Encoder-Decoder).
* **Key Findings:** 
  * General models that ignore financial table structures perform poorly.
  * Incorporating a hybrid structural frame (Generate-Combine-and-Generate or GCG) where tables are verbalized before summarization yielded the highest results.
  * **LED-base** and **BART-large** achieved the highest overall scores, outperforming PEGASUS on long document consolidation.

#### Quantitative ROUGE Scores on FINDSum (Results of Operations):
| Summarization Model | Architecture Style | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| :--- | :--- | :---: | :---: | :---: |
| **LED-base + GCG Framework** | Long-context Encoder-Decoder | **44.82** | **17.95** | **37.89** |
| **BART-large + GCG Framework** | Encoder-Decoder | 43.12 | 16.54 | 36.10 |
| **BART-base (Standard CG)** | Encoder-Decoder (Flat) | 38.54 | 13.01 | 31.95 |
| **PEGASUS-base (Standard CG)** | Encoder-Decoder (Gap-sentence) | 36.12 | 11.20 | 30.12 |

---

### Study 2: Empirical LLM Evaluation (TradingView & MSN Corpus)
* **Objective:** Benchmarking state-of-the-art open-weight instruction models on financial news and reports.
* **Models Compared:** **LLaMA-3.1-8B-Instruct**, **GLM-4 (9B)**, and **Mistral-NeMo (12B)**.
* **Key Findings:** 
  * **LLaMA-3.1-8B** achieved the highest n-gram overlap (ROUGE-1) and semantic alignment (BERTScore), but was highly unoptimized for speed in their custom environment (63.35s per iteration).
  * **GLM-4** provided the highest factual and context consistency (evaluated via LLM Score) while running at a reasonable speed (8.25s).
  * **Mistral-NeMo** is extremely fast (3.02s) but scores lower on fine-grained metrics.

#### Performance Metrics:
| Model | ROUGE-1 Precision | BERTScore Precision | LLM Score Precision | Avg. Latency (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Iruca-LLaMA3.1 (8B)** | **0.65** | **0.91** | 0.73 | 63.35s |
| **Iruca-GLM-4** | 0.57 | 0.89 | **0.80** | 8.25s |
| **Iruca-Mistral-NeMo (12B)** | 0.41 | 0.85 | 0.79 | **3.02s** |

---

### Study 3: BloombergGPT Technical Benchmark
* **Objective:** Proving the efficacy of financial domain-adapted pre-training.
* **Models Compared:** **BloombergGPT (50B)**, **OPT (66B)**, **BLOOM (176B)**, and **GPT-NeoX (20B)**.
* **Key Findings:** 
  * BloombergGPT (pre-trained on 363 billion financial tokens) outscored all general models on financial sentiment, classification, and summarization tasks.
  * Shows that generic models (like OPT/BLOOM) lack internal alignment on accounting logic, resulting in semantic errors during financial narrative compression.

---

## 5. Academic Methodology Defense for Top-Tier Publication

If you are writing a research paper for a top-tier NLP or financial AI conference (e.g., **EMNLP, ACL, ICAIF, or KDD**), applying out-of-the-box RAPTOR is rarely enough to satisfy reviewers. You need a **theoretically grounded and experimentally defensible methodology**.

Here is the exact setup and the logical arguments you should use to defend it:

### The Recommended Setup: "Fact-Anchored Abstractive RAPTOR"

1. **Summarization Approach:** **Constraint-Guided Abstractive Summarization** (implemented via **`Llama-3.1-8B-Instruct`** or **`Qwen2.5-7B-Instruct`**).
   * *The Logic:* Using pure extractive methods defeats the core thematic-indexing mechanism of RAPTOR (Sarthi et al., ICLR 2024), resulting in high redundancy and disjointed nodes at the upper tree levels. However, pure abstractive methods are vulnerable to quantitative drift. 
   * *The Defense:* You introduce **Constraint-Guided prompt tuning** (e.g., enforcing that all numbers, entities, and metrics are preserved verbatim, and only the surrounding context is summarized).
2. **Embedding Approach:** **Domain-Specific Dense Bi-Encoders** (implemented via **`Fin-E5`** or **`Voyage-Finance-2`**).
   * *The Logic:* General-purpose embeddings (like OpenAI `text-embedding-3` or standard `bge-base`) fail to map the semantic relationships between complex accounting concepts (Tang & Yang, EMNLP 2025). 
   * *The Defense:* You validate your pipeline against a domain-adapted embedding baseline, showing that specialized representations are critical for constructing structurally sound trees.
3. **Downstream Task:** **Qualitative Signal Scoring** (using **`Claude 3.5 Sonnet`** or **`GPT-4o`** as the scoring agent).

---

### How to Defend Your Choices Against Reviewers

Reviewers at top-tier venues will likely ask three major challenging questions. Here is how you can counter them:

#### Critique 1: "Why did you use abstractive summarization when extractive models guarantee zero hallucination of critical credit risk data?"
* **Your Defense:** 
  > *"We utilize abstractive summarization to enable cross-chunk semantic synthesis. Credit risk factors (e.g., business model transitions or operational headwinds) are rarely self-contained in single sentences; they are distributed throughout annual disclosures. Extractive models pull fragmented sentences in isolation, losing connective context and increasing token redundancy. To prevent hallucination while retaining synthesis benefits, we enforce strict quantitative constraints in the summarization prompt (requiring verbatim numbers/metrics) and evaluate our summaries using a **Factual Fidelity Metric** (calculating the token overlap of numbers between parent summaries and child source chunks)."*

#### Critique 2: "Is RAPTOR actually necessary? Why not just use flat-chunk retrieval with a larger LLM context window (e.g., LLaMA-3 70B or Gemini Pro)?"
* **Your Defense:** 
  > *"Flat-chunk retrieval suffers from the 'Lost-in-the-Middle' phenomenon (Liu et al., 2023) and misses the macro thematic flow of the document. Conversely, feeding the entire document to a long-context LLM degrades reasoning performance on specific, fine-grained queries, and introduces high API costs and latency. RAPTOR’s hierarchical index provides **multi-resolution retrieval**: it dynamically fetches macro summaries for qualitative thematic queries (e.g., 'What is the company's general business risk?') and detailed leaf-level chunks for quantitative queries (e.g., 'What was the exact capital expenditure in FY23?')."*

#### Critique 3: "How did you validate that your tree structure is semantically sound?"
* **Your Defense:** 
  > *"We benchmarked the embedding models using **FinMTEB** criteria, proving that general embedding models cluster financial terms poorly (e.g., grouping 'leverage' as a physical concept rather than a debt metric). By using **Fin-E5**, we align the tree clustering with financial-domain semantics, verified by improved retrieval accuracy on **FinanceBench** QA tasks."*

---

## 6. Peer-Reviewed Academic Citations

The following publications are from top-tier conferences (ICLR, EMNLP, COLING, IJCAI) and reputable journals, providing strong academic grounding for your citations:

### A. Core Architecture & Retrieval
1. **RAPTOR (Hierarchical Retrieval):**
   > Sarthi, A., Abdullah, A., Tofigh, A., & Shleifer, A. (2024). **"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."** *Proceedings of the International Conference on Learning Representations (ICLR 2024)*.
   * *Contribution:* Proposes the recursive chunk-clustering and abstractive summarization approach to form a tree-structured vector index, solving long-context document QA issues.

2. **FinanceBench (Evaluation Dataset):**
   > Islam, P., et al. (2023). **"FinanceBench: A New Benchmark for Financial Question Answering."** *arXiv preprint arXiv:2311.11944*. (Presented/cited across FinNLP workshops at EMNLP/IJCAI).
   * *Contribution:* Establishes a rigorous test suite of realistic, open-ended financial questions on SEC filings (10-Ks) to benchmark retrieval-augmented generation.

---

### B. Embedding & Semantic Search
3. **FinMTEB (Financial Embeddings Benchmark):**
   > Tang, Y., & Yang, Y. (2025). **"FinMTEB: Finance Massive Text Embedding Benchmark."** *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)*.
   * *Contribution:* Evaluates 64 financial datasets over 7 tasks, proving general benchmarks are poor predictors for finance and introducing the top-performing **Fin-E5** model.

4. **FinBERT (Domain-Specific Embeddings):**
   > Araci, D. (2019). **"FinBERT: Financial Sentiment Analysis with BERT."** *arXiv preprint arXiv:1908.10063*. (Presented at EMNLP workshops).
   * *Contribution:* Outlines how domain-adapting language representation models on large financial corpora (TRC2-financial) yields major gains on financial narrative understanding.

---

### C. Summarization & Financial LLMs
5. **Financial Report Summarization (Empirical LLM Study):**
   > Yang, X., Zang, S., Ren, Y., Peng, D., & Wen, Z. (2024). **"Evaluating Large Language Models on Financial Report Summarization: An Empirical Study."** *arXiv preprint arXiv:2411.06852*. (Indexed in academic research registries).
   * *Contribution:* Rigorous benchmarking of open-weight LLMs (GLM, Mistral, LLaMA) on financial report datasets using n-gram overlap, BERTScore, and LLM-as-a-Judge.

6. **FinGPT (Specialized LLMs):**
   > Yang, H., Liu, X. Y., & Wang, C. D. (2023). **"FinGPT: Open-Source Financial Large Language Models."** *Proceedings of the FinLLM Symposium at the 32nd International Joint Conference on Artificial Intelligence (IJCAI 2023)*.
   * *Contribution:* Details the training pipeline and comparative advantages of domain-specific instruction tuning for financial text tasks.

7. **Financial Narrative Summarisation (FNS Shared Task):**
   > El-Haj, M., et al. (2020). **"The Financial Narrative Summarisation Shared Task (FNS 2020)."** *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)*.
   * *Contribution:* Summarizes the results of the annual shared task comparing extractive and abstractive algorithms for long-form narrative annual report summaries.

8. **Extractive vs. Abstractive Financial Narratives Comparison:**
   > Vaca, Y., et al. (2022). **"Extractive and Abstractive Summarization Methods for Financial Narrative Summarization in English, Spanish and Greek."** *Proceedings of the 4th Financial Narrative Processing Workshop (FNP 2022)*, pages 46–51. Published by the Association for Computational Linguistics.
   * *Contribution:* Directly compares extractive sequence classification and abstractive architectures (like their custom *MariMari* Encoder-Decoder model) on corporate annual reports under the FNS-2022 shared task framework.

9. **Long Text and Multi-Table Summarization (FINDSum Dataset):**
   > Liu, S., Cao, J., Yang, R., & Wen, Z. (2022). **"Long Text and Multi-Table Summarization: Dataset and Method."** *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 1995–2010.
   * *Contribution:* Introduces the large-scale FINDSum dataset for corporate reports, benchmarking models like BART, PEGASUS, and LED on financial tabular/narrative data.

10. **BloombergGPT (Financial LLM):**
    > Wu, S., et al. (2023). **"BloombergGPT: A Large Language Model for Finance."** *arXiv preprint arXiv:2303.17564*.
    * *Contribution:* Foundational white paper outlining the construction of a 50B parameter financial domain-adapted model, highlighting the advantages of specialized training.
