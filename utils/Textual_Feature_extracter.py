import json
from utils.tokenizer import *
from utils.chunker import *
from utils.summarizer import *

# --- COMPANY PROMPT TEMPLATE ---
COMPANY_PROMPT_TEMPLATE = """
You are a senior corporate credit risk analyst.

Assess ONLY NON-FINANCIAL qualitative credit risk using the company's information provided.

IMPORTANT:
Do NOT evaluate leverage, profitability, liquidity, margins, or cash flow.
Those are already captured through numeric KPIs.

Focus only on BUSINESS MODEL and EARNINGS STABILITY risk.

Return your answer in VALID JSON ONLY. No explanations. No extra text.

SCORING SCALE:
1 = Very Low Risk / Strong Profile
2 = Low Risk
3 = Moderate / Balanced
4 = Elevated Risk
5 = High Risk / Weak Profile

If evidence is unclear or missing, assign 3.

Evaluate ONLY these three factors:

1. industry_business_model_risk  
(Cyclicality, disruption risk, regulation, demand sensitivity)

2. competitive_position_risk  
(Moat strength, differentiation, pricing power, durability)

3. revenue_stability_risk  
(Recurring vs project-based revenue, diversification of demand)

Also provide:

overall_qualitative_credit_risk  
(Overall non-financial qualitative credit risk score from 1–5)

Return JSON in EXACTLY this format:

{
  "industry_business_model_risk": 3,
  "competitive_position_risk": 3,
  "revenue_stability_risk": 3,
  "overall_qualitative_credit_risk": 3
}

Company Business Summary:
{summary}
"""

# --- PEER PROMPT TEMPLATE ---
PEER_PROMPT_TEMPLATE = """
You are a senior corporate credit risk analyst.

You are given a target company and a set of peer companies from the same industry.

Assess the TARGET company's NON-FINANCIAL qualitative credit risk RELATIVE TO ITS PEERS.

IMPORTANT:
Do NOT evaluate leverage, profitability, liquidity, margins, or any financial ratios.
Focus only on BUSINESS MODEL quality and EARNINGS STABILITY relative to peers.

Return your answer in VALID JSON ONLY. No explanations. No extra text.

SCORING SCALE:
1 = Much Stronger / Lower Risk than peers
2 = Slightly Stronger than peers
3 = In line with peers
4 = Weaker than peers
5 = Much Weaker / Higher Risk than peers

If comparison evidence is unclear, assign 3.

Evaluate ONLY these three factors for the TARGET company:

1. industry_business_model_risk  
(Exposure to cyclicality, regulation, disruption vs peers)

2. competitive_position_risk  
(Market position, differentiation, moat vs peers)

3. revenue_stability_risk  
(Recurring vs volatile demand vs peers)

Also provide:

overall_qualitative_credit_risk  
(Overall qualitative risk of target relative to peers)

Return JSON in EXACTLY this format:

{
  "industry_business_model_risk": 3,
  "competitive_position_risk": 3,
  "revenue_stability_risk": 3,
  "overall_qualitative_credit_risk": 3
}

TARGET COMPANY:
Name: {company_name}
Industry: {industry}

PEER COMPANIES:
{peer_block}
"""

# --- Format peers for LLM ---
def format_peers(peer_analysis):
    formatted = []
    for peer in peer_analysis["peers"]:
        formatted.append(
            f"Name: {peer.get('name')}\n"
            f"Industry: {peer.get('industry')}\n"
            f"Business Notes: {peer.get('business_summary', 'Not provided')}\n"
        )
    return "\n".join(formatted)


# --- Main function ---
def extract_qualitative_credit_features(input_data, source):
    """
    source = 0 → Use company 10-K text
    source = 1 → Use peer comparison mode
    """

    try:
        if source == 0:
            # Company Filing Mode
            full_text = input_data["full_text"]
            summary = summarize_whole_text(full_text)

            prompt = COMPANY_PROMPT_TEMPLATE.format(summary=summary)

        elif source == 1:
            # Peer Comparison Mode
            company_name = input_data["new_company"]["name"]
            industry = input_data["new_company"]["industry"]
            peer_block = format_peers(input_data)

            prompt = PEER_PROMPT_TEMPLATE.format(
                company_name=company_name,
                industry=industry,
                peer_block=peer_block
            )
        else:
            raise ValueError("source must be 0 (company) or 1 (peer mode)")

        response = generate_local(prompt, max_new_tokens=200)

        # Safe JSON extraction
        start = response.find("{")
        end = response.rfind("}") + 1
        result = json.loads(response[start:end])

    except Exception:
        # fallback neutral scores
        result = {
            "industry_business_model_risk": 3,
            "competitive_position_risk": 3,
            "revenue_stability_risk": 3,
            "overall_qualitative_credit_risk": 3
        }

    return result
