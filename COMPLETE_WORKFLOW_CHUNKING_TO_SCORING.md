# End-to-End RAPTOR Workflow: Chunking → Summarization → Scoring (0-10)

## Executive Overview

This document provides a **step-by-step walkthrough** of the entire pipeline from raw 10-K text to final credit risk score (0-10), with concrete examples at each stage.

```
Raw 10-K (400+ pages)
        ↓
[STAGE 1: EXTRACTION]
Extract Items 1, 1A, 3, 7, 11 (~50-100 pages)
        ↓
[STAGE 2: CHUNKING]
Segment into overlapping chunks (~800 tokens each)
        ↓
[STAGE 3: EMBEDDING]
Encode chunks using SBERT (dense vectors)
        ↓
[STAGE 4: HIERARCHICAL CLUSTERING] ← RAPTOR CORE
Soft-cluster with GMM → Recursive summarization → Multi-level tree
        ↓
[STAGE 5: DIMENSION SCORING]
Query hierarchy → LLM scoring with dimension-specific prompts
        ↓
[STAGE 6: AGGREGATION]
Weight 5 dimensions → Overall credit score (0-10)
        ↓
Final Output: Scored credit risk assessment with source traceability
```

---

## STAGE 1: EXTRACTION (15 minutes)

### Goal
Filter the 10-K to remove noise and keep only dense narrative sections.

### Input
- Raw 10-K filing (markdown, HTML, or text)
- Typically 2,900–3,500 lines
- ~400+ pages in PDF

### Process

#### Step 1a: Identify Section Headers
```python
import re

# Pattern matching for 10-K Item headers
item_patterns = {
    'Item 1': r'Item\s+1[\.\:]?\s+(?:Business)?',
    'Item 1A': r'Item\s+1A[\.\:]?\s+(?:Risk Factors)?',
    'Item 3': r'Item\s+3[\.\:]?\s+(?:Legal Proceedings)?',
    'Item 7': r'Item\s+7[\.\:]?\s+(?:MD&A|Management\'s Discussion)',
    'Item 11': r'Item\s+11[\.\:]?\s+(?:Executive Compensation)?',
}

# Regex finds section boundaries
match = re.search(pattern, filing_text, re.IGNORECASE | re.DOTALL)
start_line = match.start()
end_line = re.search(r'Item \d+', filing_text[match.end():]).start() + match.end()

extracted = filing_text[start_line:end_line]
```

#### Step 1b: Clean Extracted Text
```python
# Remove markdown artifacts, HTML tags, excessive whitespace
extracted = re.sub(r'<[^>]+>', '', extracted)  # Strip HTML
extracted = re.sub(r'^#+\s+', '', extracted, flags=re.MULTILINE)  # Strip markdown headers
extracted = re.sub(r'\n\s*\n+', '\n\n', extracted)  # Normalize whitespace

# Result: Clean prose ready for chunking
```

### Output

**Example: Item 7 (MD&A) Extraction from HAVA**
```
Before: 2,915 lines (full 10-K)
After:  142 lines (Item 7 only)
Quality: High signal, low noise

Extracted text:
"...Formation and operating costs are reviewed and monitored by the CODM 
to manage and forecast cash to ensure that enough capital is available 
to complete a Business Combination or similar transaction within the 
Business Combination period..."
```

---

## STAGE 2: CHUNKING (5 minutes)

### Goal
Break long narratives into overlapping chunks that preserve local context while enabling hierarchical clustering.

### Key Parameters
- **Chunk Size**: ~800 tokens (~3,200 characters)
  - Large enough to preserve context
  - Small enough to cluster meaningfully
- **Overlap**: ~200 tokens (~800 characters)
  - Prevents concept fragmentation at chunk boundaries
  - Enables RAPTOR to discover cross-chunk relationships

### Process

#### Step 2a: Token-Aware Segmentation
```python
def segment_into_chunks(text: str, chunk_size: int = 800, 
                       overlap: int = 200) -> List[str]:
    """
    Segment text into overlapping chunks.
    
    Args:
        text: Full extracted section (~5,000-20,000 characters)
        chunk_size: ~800 tokens = ~3,200 characters
        overlap: ~200 tokens = ~800 characters
        
    Returns:
        List of chunks, each ~800 tokens
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Extract chunk
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
    
    return chunks

# Example with Item 7 (MD&A) from HAVA
mda_text = """Formation and operating costs are reviewed and monitored 
by the CODM to manage and forecast cash to ensure that enough capital 
is available to complete a Business Combination or similar transaction 
within the Business Combination period. The CODM also reviews formation 
and operating costs to manage, maintain and enforce all contractual 
agreements to ensure costs are aligned with all agreements and budget. 
Formation and operating costs, as reported on the accompanying statements 
of operations, are the significant segment expenses provided to the CODM 
on a regular basis. All other segment items included in net income or loss 
are reported on the accompanying statements of operations..."""

chunks = segment_into_chunks(mda_text, chunk_size=800, overlap=200)
# Output: 3-4 chunks, each ~800 tokens with 200-token overlap
```

### Output

**Example Chunks from Item 1 (Business)**

**Chunk 0 (Tokens 0-800):**
```
"We are a blank check company incorporated as a Cayman Islands exempted 
company for the purpose of effecting a merger, share exchange, asset 
acquisition, share purchase, recapitalization, reorganization or similar 
business combination involving the Company, with one or more businesses 
or entities, which we refer to throughout this report as our 'initial 
business combination'. We have neither engaged in any operations nor 
generated any revenue to date. Based on our business activities, we are 
a 'shell company' as defined under the Securities Exchange Act of 1934..."
[continued to 800 tokens]
```

**Chunk 1 (Tokens 600-1400, with 200-token overlap):**
```
"...as defined under the Securities Exchange Act of 1934 (the 'Exchange 
Act') because we have no operations and nominal assets consisting almost 
entirely of cash. On October 24, 2025, the Company consummated its initial 
public offering (the 'IPO') of 14,500,000 units ('Units'). Each Unit consists 
of one Class A ordinary share, $0.0001 par value per share ('Class A ordinary 
shares'), and one right ('Rights') to receive of one-tenth of one Class A 
ordinary share upon the completion of the initial business combination..."
[continued to 1400 tokens total, with 200 from previous chunk]
```

---

## STAGE 3: EMBEDDING (2-3 minutes per section)

### Goal
Convert text chunks into dense vectors that capture semantic meaning, enabling clustering.

### Embedding Model
- **SBERT (MiniLM-L6-v2)**
  - Size: 22M parameters (lightweight)
  - Output: 384-dimensional vectors
  - Speed: ~5,000 chunks/minute on CPU
  - Cost: Free (open-source)

### Process

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize model (downloads once, cached locally)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed chunks
chunks = [chunk_0, chunk_1, chunk_2, ...]  # List of 20-50 chunks
embeddings = embedder.encode(chunks, show_progress_bar=True)

# Output: (n_chunks, 384) matrix
# Example: 30 chunks → (30, 384) matrix
print(f"Embeddings shape: {embeddings.shape}")  # (30, 384)
```

### Output

**Embedding Space Visualization (Conceptual):**
```
Chunk 0: "Blank check company... IPO of 14.5M units"
  ↓ SBERT
  Vector [0.23, -0.15, 0.89, ..., 0.12]  ← 384 dimensions

Chunk 1: "IPO proceeds... Trust Account structure"
  ↓ SBERT
  Vector [0.21, -0.18, 0.87, ..., 0.14]  ← Similar to Chunk 0 (high cosine similarity)

Chunk 2: "Formation costs... quarterly burn rate"
  ↓ SBERT
  Vector [-0.12, 0.34, 0.42, ..., 0.88]  ← Different from Chunk 0 (low cosine similarity)

These vectors now feed into GMM soft-clustering...
```

---

## STAGE 4: HIERARCHICAL CLUSTERING & SUMMARIZATION (2-3 hours per dimension)

### Goal
Build a multi-level hierarchical tree that organizes chunks by semantic topic, enabling RAPTOR to discover relationships and suppress redundancy.

### The RAPTOR Algorithm (Simplified)

```
LEVEL 0 (Raw Chunks)
├─ Chunk 0: "Blank check company... IPO..."
├─ Chunk 1: "IPO proceeds... Trust Account..."
├─ Chunk 2: "Formation costs... burn rate..."
├─ Chunk 3: "We seek management teams..."
└─ [... 20+ more chunks ...]

        ↓ Embed + Cluster with GMM
        
LEVEL 1 (Soft Clusters + Summaries)
├─ Cluster A: "SPAC Structure & Capital"
│   ├─ Chunks: [0, 1, 2]  (cosine similarity > 0.7)
│   ├─ LLM Summary: "Company raised $145M in IPO with $3.4M private placement.
│   │   Funds held in Trust Account invested in US Treasuries. Quarterly burn
│   │   rate ~$137K allows ~3 years runway until combination deadline."
│   └─ Cluster probability: P(Chunk 0 ∈ Cluster A) = 0.95
│
└─ Cluster B: "Acquisition Strategy & Management"
    ├─ Chunks: [3, 4, 5, ...]  (cosine similarity > 0.7)
    ├─ LLM Summary: "Management team brings PE and investment banking experience.
    │   Acquisition criteria focus on strong management, revenue visibility,
    │   and defensible market position. CEO led prior M&As at [competitor]."
    └─ Cluster probability: P(Chunk 3 ∈ Cluster B) = 0.92

        ↓ Re-embed Summaries + Re-cluster
        
LEVEL 2 (Meta-Clusters + Root Summaries)
├─ Meta-Cluster I: "SPAC FUNDAMENTALS"
│   ├─ Children: [Cluster A, Cluster B] (semantically related)
│   └─ LLM Summary: "Harvard Ave Acquisition Corp is a SPAC with $145M capital,
│       ~3-year runway, and experienced management team focused on acquiring
│       operationally strong companies with revenue visibility."
│
└─ [Meta-Cluster II, III, ... if tree grows deeper]

        ↓ Re-embed Meta-summaries + Re-cluster (repeat until convergence)

LEVEL 3 (Root Node - CONVERGENCE)
└─ Root Summary: "SPAC acquisition vehicle with strong capitalization and 
   experienced sponsor team. Key risks: combination deadline (Oct 2027),
   redemption overhang, and execution risk on target identification."
```

### Detailed Process

#### Step 4a: Soft-Cluster with GMM
```python
from sklearn.mixture import GaussianMixture

# Embed chunks (output: (30, 384) matrix)
embeddings = embedder.encode(chunks)

# Soft-cluster with GMM (probabilistic, allows overlap)
n_clusters = max(1, len(chunks) // 4)  # ~4 chunks per cluster
gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
gmm.fit(embeddings)

# Get soft assignments: P(chunk_i ∈ cluster_k)
soft_assignments = gmm.predict_proba(embeddings)  # (30, n_clusters)
hard_assignments = gmm.predict(embeddings)  # (30,) - argmax

print(f"Clusters found: {n_clusters}")
print(f"Cluster sizes: {np.bincount(hard_assignments)}")
# Output:
# Clusters found: 8
# Cluster sizes: [4 3 5 2 4 3 2 2]
```

**What Soft Clustering Does:**
```
Example: "Complex legal dispute over unpaid contracts"

Hard clustering (K-Means):
  → Belongs to CLUSTER_5 (Regulatory Risk) only
  
Soft clustering (GMM):
  → P(CLUSTER_5 | Regulatory Risk) = 0.65
  → P(CLUSTER_3 | Revenue Risk) = 0.28  ← Also relevant!
  → P(CLUSTER_7 | Operational Risk) = 0.07

RAPTOR captures that this concept is relevant to BOTH regulatory AND revenue risk,
not just one or the other.
```

#### Step 4b: LLM Summarization with Dimension Context
```python
from anthropic import Anthropic

def summarize_cluster(cluster_chunks: List[str], 
                     dimension: str = "REVENUE_STABILITY") -> str:
    """
    Summarize a cluster of chunks using Claude.
    Dimension-specific prompting ensures focused extraction.
    """
    
    # Combine chunks
    cluster_text = "\n\n---\n\n".join(cluster_chunks)
    
    # Dimension-specific prompt
    prompt = f"""
You are analyzing a 10-K section for credit risk assessment.

DIMENSION: {dimension}
FOCUS: {"Recurring vs. volatile revenue, customer concentration, churn" 
        if dimension == "REVENUE_STABILITY" else "..."}

CLUSTER TEXT:
{cluster_text}

Provide a concise 150-200 word summary highlighting:
1. Key risk signals relevant to {dimension}
2. Specific metrics, customer names, or percentages mentioned
3. Uncertainty or red flags
4. Source traceability (paragraph references)

Be specific. Avoid generic statements.
"""
    
    client = Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Example: Summarize a cluster about revenue composition
cluster_chunks = [
    "Subscriptions represented 67% of FY2025 revenue, up from 45% in FY2024...",
    "Enterprise contracts have an average 3-year term with 92% renewal rate...",
    "However, SMB churn accelerated to 8% (from 5% historically)...",
]

summary = summarize_cluster(cluster_chunks, dimension="REVENUE_STABILITY")

# Output:
"""
The company is successfully transitioning to a subscription-based model, 
with recurring revenue now representing 67% of total revenue (up from 45% 
prior year). Enterprise segment shows strong fundamentals: 3-year contract 
terms and 92% renewal rates indicate high revenue stickiness. However, SMB 
segment exhibits concerning churn acceleration to 8% annually (from 5%), 
suggesting economic sensitivity in this customer segment. Risk signal: If 
SMB churn continues to accelerate, it could offset enterprise segment gains.
[Source: Item 7, MD&A, Revenue by Segment]
"""
```

#### Step 4c: Recursive Re-embedding & Re-clustering
```python
def build_raptor_hierarchy(chunks: List[str], 
                          dimension: str,
                          max_levels: int = 3) -> Dict:
    """
    Recursively build RAPTOR hierarchy.
    """
    
    current_chunks = chunks.copy()
    hierarchy = {}
    
    for level in range(max_levels):
        print(f"\n--- LEVEL {level} ---")
        print(f"Processing {len(current_chunks)} chunks...")
        
        # Step 1: Embed current chunks
        embeddings = embedder.encode(current_chunks)
        
        # Step 2: Determine clustering level
        n_clusters = max(1, len(current_chunks) // 4)
        n_clusters = min(n_clusters, 10)  # Cap at 10
        
        # Step 3: Soft-cluster
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(embeddings)
        assignments = gmm.predict(embeddings)
        
        # Step 4: Summarize each cluster
        next_level_summaries = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, a in enumerate(assignments) if a == cluster_id]
            if not cluster_indices:
                continue
            
            cluster_text = [current_chunks[i] for i in cluster_indices]
            summary = summarize_cluster(cluster_text, dimension)
            next_level_summaries.append(summary)
        
        # Store this level
        hierarchy[f'level_{level}'] = {
            'chunks': current_chunks,
            'n_clusters': n_clusters,
            'summaries': next_level_summaries,
        }
        
        print(f"Generated {len(next_level_summaries)} summaries")
        
        # Check convergence
        if len(next_level_summaries) <= 1:
            print(f"Converged at level {level}")
            break
        
        # Prepare for next level (summaries become new chunks)
        current_chunks = next_level_summaries
    
    return hierarchy

# Build hierarchy for REVENUE_STABILITY dimension
hierarchy_revenue = build_raptor_hierarchy(
    chunks=item7_chunks,
    dimension='REVENUE_STABILITY',
    max_levels=3
)

# Output structure:
# hierarchy_revenue = {
#     'level_0': {'chunks': [30 raw chunks], 'n_clusters': 8, 'summaries': [8 summaries]},
#     'level_1': {'chunks': [8 summaries], 'n_clusters': 2, 'summaries': [2 meta-summaries]},
#     'level_2': {'chunks': [2 meta-summaries], 'n_clusters': 1, 'summaries': [1 root summary]},
# }
```

### Output: RAPTOR Hierarchy for REVENUE_STABILITY

```
HIERARCHY (3 Levels):

LEVEL 0 (Raw Chunks):
├─ Chunk 0: "Subscriptions represented 67%..."
├─ Chunk 1: "Enterprise contracts have 3-year terms..."
├─ Chunk 2: "SMB churn accelerated to 8%..."
├─ Chunk 3: "Customer acquisition cost..."
└─ [... 26 more chunks]

LEVEL 1 (Clustered Summaries):
├─ Summary 0: "Revenue mix is shifting toward subscriptions (67%), driven by
              enterprise segment strength (3-yr terms, 92% renewal). SMB
              segment shows weakness with 8% churn acceleration. Overall,
              recurring revenue base is strengthening but SMB segment risk
              is emerging."
├─ Summary 1: "Customer concentration moderate: Top 5 customers = 32% of revenue
              (up from 28% prior year). Largest single customer = 8%.
              Concentration trend is concerning; if accelerates to top 5 = 40%+,
              revenue stability risk increases materially."
├─ Summary 2: "Backlog/pipeline visibility: Average contract 3 years provides
              forward revenue visibility. However, downward revisions to
              FY2025 guidance (-2% from prior guide) suggest pipeline
              challenges. Pipeline conversion risk is elevated."
└─ [... 5 more summaries]

LEVEL 2 (Meta-Summaries):
├─ Meta-Summary 0: "REVENUE QUALITY ASSESSMENT: Company transitioning to
                   subscription model (67% recurring) with strong enterprise
                   retention (92%) but emerging SMB weakness (8% churn).
                   Customer concentration increasing (top 5 = 32%) is concerning.
                   Contract length (3 yrs) provides visibility but guidance
                   misses suggest pipeline stress. Overall: Revenue base
                   strengthening but concentration risk and SMB churn offset gains."
└─ Meta-Summary 1: "REVENUE HEADROOM & CAPACITY: $X annual run-rate;
                   3-year contracts provide forward visibility; cash position
                   allows for growth investments. No imminent solvency risk,
                   but growth guidance cuts indicate market headwinds."

LEVEL 3 (Root Node - CONVERGENCE):
└─ Root Summary: "Company has strong recurring revenue foundation (67% subscriptions,
                 92% enterprise renewal) but faces emerging risks from SMB churn
                 (8% vs 5% prior), customer concentration (top 5 = 32%, rising),
                 and pipeline challenges (guidance misses). Revenue stability is
                 GOOD but DETERIORATING. Watch customer concentration and SMB
                 churn trend."
```

---

## STAGE 5: DIMENSION SCORING (30 minutes for all 5 dimensions)

### Goal
Query the RAPTOR hierarchy to extract dimension-specific signals, then score on 0–10 scale with risk/mitigation factors.

### Process

#### Step 5a: Query Hierarchy for Dimension Context
```python
def query_hierarchy(hierarchy: Dict, dimension: str) -> str:
    """
    Extract most relevant context from RAPTOR hierarchy for scoring.
    Uses deepest level (highest abstraction) for conciseness.
    """
    
    # Find deepest level
    deepest_summaries = None
    for level in range(10, -1, -1):
        if f'level_{level}' in hierarchy and hierarchy[f'level_{level}'].get('summaries'):
            deepest_summaries = hierarchy[f'level_{level}']['summaries']
            break
    
    if not deepest_summaries:
        deepest_summaries = hierarchy['level_0']['chunks']
    
    # Combine summaries
    context = "\n\n---\n\n".join(deepest_summaries)
    return context

# Example
context = query_hierarchy(hierarchy_revenue, 'REVENUE_STABILITY')
# Returns: ~500 words of highest-level summaries capturing all revenue dynamics
```

#### Step 5b: Dimension-Specific Scoring Prompt
```python
def score_dimension(hierarchy: Dict, dimension: str, 
                   item_source: str = "Item 7") -> Tuple[float, Dict]:
    """
    Score a dimension 0-10 based on RAPTOR hierarchy.
    """
    
    # Query hierarchy for context
    context = query_hierarchy(hierarchy, dimension)
    
    # Dimension-specific prompt
    scoring_prompt = f"""
You are a senior credit analyst scoring corporate credit risk.

DIMENSION: {dimension}
SOURCE: {item_source}

HIERARCHY CONTEXT (extracted via RAPTOR):
{context}

SCORE THIS DIMENSION ON 0–10 SCALE:
- 9–10: Minimal risk (strong position, high confidence)
- 7–8: Low risk (well-managed, clear visibility)
- 5–6: Moderate risk (balanced risk/mitigation)
- 3–4: High risk (significant concerns, limited mitigation)
- 0–2: Critical risk (major concerns, unmitigated exposure)

RESPOND WITH JSON:
{{
    "score": <0-10 float>,
    "risk_factors": [<3-5 specific risks>],
    "mitigating_factors": [<2-3 specific mitigations>],
    "key_metrics": [<specific values: "67% recurring", "top 5 = 32%", etc>],
    "source_citations": [<"Item X, Section Y, quote">, ...],
    "confidence": "<HIGH|MEDIUM|LOW>",
    "narrative": "<2-sentence executive summary>"
}}
"""
    
    client = Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    
    response_text = message.content[0].text
    
    # Parse JSON
    import json
    import re
    
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
    else:
        json_text = response_text
    
    scored = json.loads(json_text)
    score = float(scored['score'])
    
    return score, scored

# Score REVENUE_STABILITY
score_revenue, details_revenue = score_dimension(
    hierarchy_revenue,
    dimension='REVENUE_STABILITY',
    item_source='Item 7'
)

print(f"REVENUE_STABILITY Score: {score_revenue}/10")
print(f"Confidence: {details_revenue['confidence']}")
print(f"Narrative: {details_revenue['narrative']}")
```

### Example Scoring Output

**REVENUE STABILITY DIMENSION**

```json
{
    "score": 7.3,
    "risk_factors": [
        "Customer concentration increasing: top 5 customers grew from 28% to 32%",
        "SMB churn accelerated to 8% (from 5%) due to economic sensitivity",
        "Guidance misses (-2% variance) suggest pipeline challenges",
        "Subscription model still ramping (67% vs 45% prior year) = execution risk"
    ],
    "mitigating_factors": [
        "67% recurring revenue base (up from 45%) provides stability",
        "Enterprise segment strong: 3-year contracts, 92% renewal rate",
        "Contract length provides 3-year forward revenue visibility",
        "No single-customer concentration (largest = 8% only)"
    ],
    "key_metrics": [
        "Recurring revenue: 67% (up from 45% prior year)",
        "Top 5 customers: 32% of revenue (up from 28%)",
        "Top single customer: 8% of revenue",
        "Enterprise churn: 3% (stable)",
        "SMB churn: 8% (up from 5%)",
        "Avg contract length: 3 years",
        "Enterprise renewal rate: 92%"
    ],
    "source_citations": [
        "Item 7, MD&A, Revenue by Type: '67% of revenue from subscriptions'",
        "Item 7, MD&A, Customer Concentration: 'Top 5 customers = 32%'",
        "Item 7, MD&A, Segment Analysis: 'SMB churn accelerated to 8%'",
        "Item 1, Business: 'Enterprise contracts average 3 years'"
    ],
    "confidence": "HIGH",
    "narrative": "Revenue base is strengthening with shift to recurring subscriptions (67%) 
                and strong enterprise retention (92% renewal), but emerging risks include 
                rising customer concentration (top 5 = 32%, up from 28%) and SMB churn 
                acceleration (8% vs 5%). Overall: GOOD but DETERIORATING trend."
}
```

### Scoring Rubric Details

For **REVENUE_STABILITY** specifically:

```
DIMENSION: REVENUE_STABILITY
Measures: Recurring %, Customer Concentration, Churn, Contract Length

RUBRIC:

9–10 (MINIMAL RISK):
  ✓ 80%+ recurring revenue
  ✓ Top customer <15% of revenue
  ✓ Churn <3% annually
  ✓ Multi-year contracts with high renewal rates (90%+)
  ✓ Stable or declining concentration
  Example: SaaS company with 85% recurring, top 5 = 20%, 2% churn

7–8 (LOW RISK):
  ✓ 60–80% recurring revenue
  ✓ Top customer 15–25% of revenue
  ✓ Churn 3–8% annually
  ✓ 2–3 year contracts with 85%+ renewal
  ✓ Stable concentration
  Example: Company with 70% recurring, top 5 = 28%, 6% churn (stable)

5–6 (MODERATE RISK):
  ✓ 40–60% recurring revenue
  ✓ Top customer 25–40% of revenue
  ✓ Churn 8–15% annually
  ✓ 1–2 year contracts with 75%+ renewal
  ✓ Rising concentration trend
  Example: Company with 55% recurring, top 5 = 35%, 10% churn (rising)

3–4 (HIGH RISK):
  ✓ <40% recurring revenue
  ✓ Top customer 40–60% of revenue
  ✓ Churn >15% annually
  ✓ Annual contracts, declining renewal rates
  ✓ Accelerating concentration
  Example: Company with 30% recurring, top 3 = 50%, 18% churn (accelerating)

0–2 (CRITICAL RISK):
  ✓ <20% recurring revenue
  ✓ Top customer >60% of revenue
  ✓ Churn >25% annually
  ✓ Transactional model, no contracts
  ✓ Concentration at extreme levels
  Example: Company dependent on 1-2 large customers, high churn

SCORING LOGIC FOR EXAMPLE:
  Recurring: 67% → +2.0 points (good, but still ramping)
  Concentration: Top 5 = 32%, rising → -1.5 points (concerning trend)
  Churn: Enterprise 3% (good, +1.0), SMB 8% (bad, -0.5) → Mixed
  Contracts: 3 years, 92% renewal → +1.5 points (strong)
  Trend: Improving mix offset by rising concentration & SMB issues → Neutral
  
  Base: 5.0
  Adjustments: +2.0 - 1.5 + 0.5 + 1.5 = +2.5
  Final Score: 7.5
  
  But SMB churn accelerating is red flag → Adjust down to 7.3
```

---

## STAGE 6: AGGREGATION (5 minutes)

### Goal
Combine 5 dimension scores into weighted overall credit score.

### Process

#### Step 6a: Score All 5 Dimensions

```python
def score_all_dimensions(hierarchies: Dict[str, Dict]) -> Dict[str, Tuple[float, Dict]]:
    """
    Score all 5 dimensions from their respective hierarchies.
    """
    
    dimensions = [
        ('INDUSTRY_BUSINESS', 'Item 1'),
        ('REVENUE_STABILITY', 'Item 7'),
        ('OPERATIONAL_ASSET', 'Item 1'),
        ('MANAGEMENT_EXECUTION', 'Item 1'),
        ('REGULATORY_LICENSING', 'Item 1A'),
    ]
    
    all_scores = {}
    
    for dimension, item_source in dimensions:
        if dimension not in hierarchies:
            print(f"⚠ Hierarchy for {dimension} not found")
            continue
        
        score, details = score_dimension(
            hierarchies[dimension],
            dimension,
            item_source
        )
        
        all_scores[dimension] = (score, details)
        print(f"✓ {dimension}: {score}/10 ({details['confidence']})")
    
    return all_scores

# Score all dimensions
all_scores = score_all_dimensions(hierarchies)

# Output:
# ✓ INDUSTRY_BUSINESS: 7.2/10 (HIGH)
# ✓ REVENUE_STABILITY: 7.3/10 (HIGH)
# ✓ OPERATIONAL_ASSET: 6.8/10 (MEDIUM)
# ✓ MANAGEMENT_EXECUTION: 7.5/10 (HIGH)
# ✓ REGULATORY_LICENSING: 5.5/10 (MEDIUM)
```

#### Step 6b: Apply Dimension Weights

```python
# Weights based on credit risk importance
DIMENSION_WEIGHTS = {
    'INDUSTRY_BUSINESS': 0.20,           # 20% - Sector fundamentals
    'REVENUE_STABILITY': 0.25,           # 25% - Most important (ability to repay)
    'OPERATIONAL_ASSET': 0.20,           # 20% - Production capability
    'MANAGEMENT_EXECUTION': 0.20,        # 20% - Strategy delivery
    'REGULATORY_LICENSING': 0.15,        # 15% - Legal/regulatory risk
}

# Compute weighted score
overall_score = 0.0
for dimension, (score, details) in all_scores.items():
    weight = DIMENSION_WEIGHTS[dimension]
    overall_score += score * weight
    print(f"{dimension:25s}: {score:5.1f} × {weight:.0%} = {score * weight:5.2f}")

print(f"\n{'=' * 60}")
print(f"Overall Credit Score: {overall_score:.1f}/10")
print(f"{'=' * 60}")

# Output:
# INDUSTRY_BUSINESS        :   7.2 × 20% =  1.44
# REVENUE_STABILITY        :   7.3 × 25% =  1.83
# OPERATIONAL_ASSET        :   6.8 × 20% =  1.36
# MANAGEMENT_EXECUTION     :   7.5 × 20% =  1.50
# REGULATORY_LICENSING     :   5.5 × 15% =  0.83
# ============================================================
# Overall Credit Score: 7.0/10
```

#### Step 6c: Map Score to Credit Grade

```python
def score_to_credit_grade(score: float) -> Tuple[str, str, str]:
    """
    Map 0-10 score to credit grade and interpretation.
    """
    
    if score >= 8.5:
        return "AAA/AA", "Minimal Risk", "Excellent credit quality"
    elif score >= 7.5:
        return "A", "Low Risk", "Good credit quality"
    elif score >= 6.5:
        return "BBB", "Moderate Risk", "Adequate credit quality"
    elif score >= 5.0:
        return "BB", "High Risk", "Speculative credit quality"
    elif score >= 3.0:
        return "B", "Very High Risk", "Highly speculative"
    else:
        return "CCC/C", "Critical Risk", "Distressed/Default risk"

grade, risk_level, interpretation = score_to_credit_grade(overall_score)

print(f"""
CREDIT ASSESSMENT SUMMARY
═════════════════════════════════════════

Overall Score:      {overall_score:.1f}/10
Credit Grade:       {grade}
Risk Level:         {risk_level}
Interpretation:     {interpretation}

Dimensional Breakdown:
├─ Industry & Business:    7.2/10 (Low risk - moderate competitive position)
├─ Revenue Stability:       7.3/10 (Low risk - strong recurring base)
├─ Operational & Asset:     6.8/10 (Moderate risk - supply chain concentration)
├─ Management Execution:    7.5/10 (Low risk - clear strategy, some guidance misses)
└─ Regulatory & Licensing:  5.5/10 (High risk - pending litigation, SPAC complexity)

Key Strengths:
✓ 67% recurring revenue with 92% enterprise renewal rate
✓ Experienced management team with PE/investment banking background
✓ 3-year contract visibility
✓ Strong cash position ($146M trust account)

Key Concerns:
⚠ Rising customer concentration (top 5 = 32%, up from 28%)
⚠ SMB segment churn accelerating (8% vs 5%)
⚠ SPAC combination deadline creates redemption overhang
⚠ Unresolved litigation risks

Recommendation:
→ INVESTMENT GRADE CREDIT (7.0/10)
  Suitable for institutional portfolios with moderate risk tolerance
  Monitor: customer concentration trends, SMB churn progression, litigation outcomes
""")
```

---

## COMPLETE END-TO-END EXAMPLE

### Input: HAVA 10-K

Let me walk through scoring one dimension completely.

#### Scenario: Score MANAGEMENT_EXECUTION (Item 1 + Item 7)

**Raw Extraction (Stage 1):**
```
Item 1 excerpt:
"...management team's proprietary network of relationships with corporate 
executives, private equity, venture and growth capital funds, investment 
banking firms and consultants...

Mr. Sung Hyuk Lee, our Chief Executive Officer and Chairman, has extensive 
experience in private equity, corporate finance, and financial advisory. 
Mr. Choi, our Chief Financial Officer and Director, is an experienced 
investment management professional..."

Item 7 excerpt:
"Formation and operating costs are reviewed and monitored by the CODM to 
manage and forecast cash to ensure that enough capital is available to 
complete a Business Combination..."
```

**Chunking (Stage 2):**
```
Chunk A (800 tokens):
"...management team's proprietary network of relationships...
Mr. Sung Hyuk Lee...extensive experience in private equity...
Mr. Choi...experienced investment management professional..."

Chunk B (800 tokens):
"Formation and operating costs are reviewed and monitored...
cash to ensure that enough capital is available...
Formation and operating costs...significant segment expenses..."
```

**Embedding (Stage 3):**
```
Chunk A → [0.12, -0.34, 0.78, ..., 0.45]  (384-dim vector)
Chunk B → [-0.08, 0.22, 0.51, ..., 0.61]  (384-dim vector)

Cosine similarity: 0.42 (moderately related; different topics)
```

**Hierarchical Clustering (Stage 4):**
```
LEVEL 0: [Chunk A, Chunk B, Chunk C, ...]  (10 chunks total)

LEVEL 1 Clustering:
├─ Cluster 1: Management Background
│   ├─ Chunk A (P=0.95)
│   └─ Summary: "Management team brings extensive PE and investment banking 
│       experience. CEO led M&As at prior firm; CFO has investment mgmt background. 
│       Team competencies well-aligned with SPAC acquisition strategy."
│
└─ Cluster 2: Capital Management & Operations
    ├─ Chunk B (P=0.98)
    └─ Summary: "Company maintains $146M capital in trust. Quarterly burn 
        rate ~$137K allows ~3-year runway to Oct 2027 combination deadline. 
        Capital management discipline evident in cost structure."

LEVEL 2 (Convergence):
└─ Root Summary: "Management team demonstrates both relevant experience 
   (PE/investment banking background) and operational discipline (capital 
   management to deadline). Key risks: management not obligated to remain 
   post-acquisition; execution depends on finding suitable target and retaining 
   team."
```

**Dimension Scoring (Stage 5):**
```
Query: Context from hierarchy
→ "Management team brings extensive PE/investment banking experience...
   Capital management discipline evident...
   Management not obligated to remain post-acquisition..."

LLM Scoring Prompt:
"Score MANAGEMENT_EXECUTION on 0-10 based on above context..."

Output:
{
    "score": 7.5,
    "risk_factors": [
        "Management team not contractually obligated to remain post-acquisition",
        "Execution depends entirely on successful target identification",
        "Track record of current management team as cohesive SPAC unit not yet proven"
    ],
    "mitigating_factors": [
        "CEO & CFO bring significant PE/investment banking experience",
        "Acquisition criteria clearly defined and well-articulated",
        "Capital management demonstrates operational discipline"
    ],
    "key_metrics": [
        "CEO experience: 15+ years in private equity",
        "CFO background: investment management professional",
        "Capital runway: ~3 years (to Oct 2027 deadline)",
        "Quarterly burn: ~$137K"
    ],
    "confidence": "HIGH",
    "narrative": "Management team possesses relevant experience (PE, M&A) and demonstrates 
                capital discipline, but execution risk remains high due to unproven SPAC 
                track record and lack of contractual commitment post-acquisition."
}
```

**Aggregation (Stage 6):**
```
MANAGEMENT_EXECUTION Score: 7.5/10 × 20% weight = 1.50 points

Combined with other dimensions:
  INDUSTRY_BUSINESS:       7.2 × 20% = 1.44
  REVENUE_STABILITY:       7.3 × 25% = 1.83
  OPERATIONAL_ASSET:       6.8 × 20% = 1.36
  MANAGEMENT_EXECUTION:    7.5 × 20% = 1.50  ← Our dimension
  REGULATORY_LICENSING:    5.5 × 15% = 0.83
  ────────────────────────────────────────
  OVERALL SCORE:           7.0/10 = Grade A (Low Risk)
```

---

## RAPTOR ADVANTAGES OVER FLAT RAG

### Problem: Flat RAG Loses Context

```
Raw 10-K chunks (flat):
└─ Chunk 0: "Top 5 customers = 32% of revenue"
└─ Chunk 1: "Cyclical industry subject to economic cycles"
└─ Chunk 2: "Signed long-term contracts provide visibility"

Retrieval for "revenue risk":
  → Returns Chunk 0 (customer concentration)
  → Returns Chunk 1 (cyclicality)
  → Returns Chunk 2 (contracts)

Problem: Chunks isolated; analyst must manually connect:
  "Customer concentration (Chunk 0) + cyclicality (Chunk 1) 
   - long-term contracts (Chunk 2) = moderate revenue risk"
```

### Solution: RAPTOR Hierarchical Clustering

```
RAPTOR hierarchy (soft-clustered):
└─ Level 0: [Chunk 0, Chunk 1, Chunk 2, ...]
└─ Level 1:
    ├─ Cluster A: "Revenue Composition & Stability"
    │   └─ Chunks: [0, 2] (soft: P=0.88, 0.95)
    │   └─ Cluster Phrase: "Customer concentration in cyclical industry 
    │       partially offset by long-term contracts"
    │
    └─ Cluster B: "Economic Sensitivity"
        └─ Chunks: [1] (soft: P=0.92)
        └─ Cluster Phrase: "Revenue exposed to economic downturns"

Retrieval for "revenue risk":
  → Returns Level 1 Cluster A: 
    "Customer concentration (32%) in cyclical industry, partially mitigated 
     by long-term contracts. Risk level: MODERATE."

RAPTOR automatically linked the concepts!
No manual synthesis needed.
```

---

## Summary Table: Chunking → RAPTOR → Scoring

| Stage | Input | Process | Output | Time |
|-------|-------|---------|--------|------|
| 1: Extract | Raw 10-K (400+ pages) | Regex section extraction + text cleaning | Dense sections (~50-100 pages) | 15 min |
| 2: Chunk | Dense sections | Segment into 800-token overlapping chunks | 20-50 chunks | 5 min |
| 3: Embed | Chunks | SBERT encoding | (n_chunks, 384) vectors | 2-3 min |
| 4: Cluster | Embeddings | GMM soft-clustering → LLM summarization → Recursion | Multi-level hierarchy (3-4 levels) | 120 min |
| 5: Score | Hierarchy | Dimension-specific prompt + LLM scoring | 5 dimension scores (0-10 each) | 30 min |
| 6: Aggregate | 5 scores | Weighted average | Overall credit score (0-10) | 5 min |
| **Total** | | | **Scored Assessment** | **3-4 hours** |

---

## Key Takeaways

1. **Chunking**: ~800 tokens with 200-token overlap preserves local context while enabling meaningful clustering

2. **RAPTOR Magic**: Soft-clustering (GMM) + recursive summarization discovers relationships that flat RAG misses
   - "Customer concentration" + "Cyclical industry" → "High revenue risk"
   - Automatic, no manual synthesis

3. **Scoring**: Dimension-specific prompts ensure focused extraction; LLM sees hierarchy context, not raw chunks

4. **Source Traceability**: Every score traces back to source paragraph → Audit compliance

5. **Scale**: 3-4 hours per company for full pipeline; improves dramatically with caching/optimization

6. **Validation**: Weighted dimensions reflect credit importance (Revenue Stability = 25%, most critical)

This is the complete flow from raw 10-K to final credit score (0-10).

