# 10-K Section to CRED-RAPTOR Dimension Mapping

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DENSE 10-K NARRATIVE SECTIONS                    │
│                      (for RAPTOR extraction)                        │
└─────────────────────────────────────────────────────────────────────┘

Item 1: Business                    Item 1A: Risk Factors
┌─────────────────────────┐        ┌──────────────────────────┐
│ § Business Description  │        │ § Industry/Market Risks  │
│ § Competitive Position  │◄──────►│ § Operational Risks      │
│ § Strategy & Criteria   │        │ § Management Risks       │
│ § Operations & Assets   │        │ § Regulatory Risks       │
│ § Management Background │        │ § Customer Risks         │
│ § Properties            │        │ § Financial Risks        │
└─────────────────────────┘        └──────────────────────────┘

Item 7: MD&A                        Item 3: Legal Proceedings
┌─────────────────────────┐        ┌──────────────────────────┐
│ § Revenue Analysis      │        │ § Pending Litigation     │
│ § Customer Dynamics     │◄──────►│ § Regulatory Actions     │
│ § Operating Changes     │        │ § Settlement Status      │
│ § Capital Allocation    │        │ § Estimated Exposure     │
│ § Result Variances      │        │ § Timeline to Resolution │
│ § Liquidity & Resources │        └──────────────────────────┘
└─────────────────────────┘

Item 11: Exec Compensation
┌──────────────────────────┐
│ § Pay-for-Performance   │
│ § Incentive Alignment   │
│ § Clawback Provisions   │
│ § Executive Turnover    │
└──────────────────────────┘
         ▲
         │
         └──────────────────┐
                            │
                     ┌──────▼─────────┐
                     │   Management   │
                     │   Execution    │
                     │   Dimension    │
                     └────────────────┘
```

---

## Detailed Dimension ↔ Section Mapping

### 1. INDUSTRY & BUSINESS Dimension
**Score: What is the company's competitive position and sector risk?**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDUSTRY & BUSINESS                          │
│              (Competitive Positioning, Cyclicality)                 │
└─────────────────────────────────────────────────────────────────────┘

PRIMARY SOURCE SECTIONS:
├─ Item 1: Business → General & Strategy
│  ├─ Competitive advantages / differentiation
│  ├─ Market position & share
│  ├─ Industry structure (fragmented, concentrated, etc.)
│  ├─ Supplier & customer concentration
│  └─ Product/service diversification
│
├─ Item 1A: Risk Factors → Industry Risks
│  ├─ "Competition from [larger/better-resourced] players"
│  ├─ "Industry consolidation could disadvantage us"
│  ├─ "Cyclical nature of [sector]"
│  ├─ "Commodity pricing pressure"
│  └─ "Market disruption from [technology/regulation]"
│
└─ Item 7: MD&A → Revenue by Segment
   ├─ Year-over-year segment performance
   ├─ Geographic revenue trends
   └─ Product/service mix shifts

RAPTOR EXTRACTION:
├─ Macro theme: "Industry is [consolidating/fragmenting/stable]"
│  └─ Supported by: specific company mentions, deal activity, pricing trends
├─ Macro theme: "Company has [strong/weak] competitive position"
│  └─ Supported by: market share, differentiation, customer feedback
└─ Macro theme: "Sector faces [cyclical/secular] headwinds"
   └─ Supported by: demand trends, capex cycles, commodity price correlations

SCORING RUBRIC:
┌──────────────────────────────────────────────────────────────┐
│ 9-10: | Market leader, defensible moat, non-cyclical demand  │
│ 7-8:  | Stable competitive position, moderate cyclicality    │
│ 5-6:  | Competitive but not differentiated, cyclical sector  │
│ 3-4:  | Weak position, commoditized, severe cyclicality      │
│ 0-2:  | Declining industry, existential threats              │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (Technology SaaS Company):
  Item 1: "We compete on ease-of-use and customer support. 
           Market is fragmented with 200+ competitors. 
           We have 15% market share in [vertical]."
  
  Item 1A: "Software industry is characterized by rapid change. 
            Larger competitors could bundle competing products."
  
  Item 7:  "SaaS segment grew 23% YoY (52% of revenue). 
            Professional services declined 8% YoY (48% of revenue)."
  
  → RAPTOR clusters these into:
    MACRO: "Software is increasingly fragmented but consolidating"
      └─ MICRO: Company differentiates on support; faces bundling risk
    MACRO: "Revenue mix shifting toward SaaS (higher margin)"
      └─ MICRO: Professional services declining (execution risk)
    SCORE: 7.5 (good position, but cyclical tech sector)
```

---

### 2. REVENUE STABILITY Dimension
**Score: Are revenues recurring and predictable?**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REVENUE STABILITY                              │
│       (Recurring vs. Volatile, Customer Concentration, Churn)       │
└─────────────────────────────────────────────────────────────────────┘

PRIMARY SOURCE SECTIONS:
├─ Item 7: MD&A → Revenue Recognition & Trends
│  ├─ Breakdown of revenue by type (subscription, transactional, etc.)
│  ├─ Largest customers and concentration (top 5, top 10)
│  ├─ Customer acquisition cost and lifetime value
│  ├─ Churn rates and retention trends
│  ├─ Contract duration and renewal rates
│  ├─ Backlog and forward visibility
│  └─ Significant customer wins/losses
│
├─ Item 1: Business → Customer & Market
│  ├─ Customer diversification strategy
│  ├─ Long-term contracts or ongoing relationships
│  ├─ Customer concentration risks
│  └─ Market segment dependencies
│
├─ Item 1A: Risk Factors → Revenue Risks
│  ├─ "Dependence on a limited number of customers"
│  ├─ "Loss of [named] major customer would significantly impact revenue"
│  ├─ "Customer churn in [segment]"
│  ├─ "Contract renegotiation or non-renewal risks"
│  └─ "Seasonal or cyclical revenue patterns"
│
└─ Item 3: Legal Proceedings
   └─ Customer disputes, contract termination litigation

RAPTOR EXTRACTION:
├─ Recurring revenue indicator: "X% of revenue from subscriptions, 
│  Y% from services, Z% from one-time licenses"
├─ Customer concentration risk: "Top 10 customers = A%; Top 3 = B%; 
│  Top single customer = C%"
├─ Churn signal: "Customer churn accelerated to X% (from Y% prior year) 
│  due to [reason]"
├─ Contract visibility: "Average contract length = X years; 
│  Renewal rate = Y%; Backlog = $Z"
└─ Customer trend: "We added/lost [X] customers; 
   reasons: [economic, competitive, product]"

SCORING RUBRIC:
┌──────────────────────────────────────────────────────────────┐
│ 9-10: | 80%+ recurring; top customer <15%; churn <3% (stable) │
│ 7-8:  | 60-80% recurring; top customer 15-25%; churn 3-8%    │
│ 5-6:  | 40-60% recurring; top customer 25-40%; churn 8-15%   │
│ 3-4:  | <40% recurring; top customer 40-60%; churn >15%      │
│ 0-2:  | Highly volatile; customer concentration >60%; churn  │
│        | accelerating; major customer loss imminent           │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (B2B SaaS):
  Item 7: "Subscriptions represent 78% of revenue (up from 65% in prior year).
           Top 5 customers = 32% of revenue (up from 28%).
           Customer churn = 6% annually (down from 8%).
           Average contract term = 3 years with 94% renewal rate."
  
  Item 1A: "Loss of [major customer] who represents 8% of revenue 
            would significantly impact growth trajectory."
  
  Item 3:  "No material customer disputes pending."
  
  → RAPTOR clusters:
    MACRO: "Subscription model creating revenue stability"
      └─ MICRO: 78% recurring; 3-yr terms; 94% renewal → good stickiness
    MACRO: "Increasing customer concentration"
      └─ MICRO: Top 5 grew from 28% to 32%; risk if concentration continues
    MACRO: "Churn improving due to [product enhancement/market demand]"
      └─ MICRO: 6% churn is healthy; downward trend positive
    SCORE: 7.8 (stable but watch concentration)
```

---

### 3. OPERATIONAL & ASSET QUALITY Dimension
**Score: Can the company produce goods/services reliably?**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  OPERATIONAL & ASSET QUALITY                        │
│      (Supply Chain, Facilities, R&D, Capital Intensity)             │
└─────────────────────────────────────────────────────────────────────┘

PRIMARY SOURCE SECTIONS:
├─ Item 1: Business → Operations & Properties
│  ├─ Manufacturing/service delivery locations
│  ├─ Geographic concentration of operations
│  ├─ Supplier/vendor dependencies
│  ├─ Key person or single-facility risks
│  ├─ Production capacity utilization
│  ├─ R&D investments and product pipeline
│  ├─ Technology infrastructure criticality
│  └─ Outsourcing dependencies
│
├─ Item 2: Properties
│  ├─ Facilities owned vs. leased
│  ├─ Facility age and condition
│  ├─ Environmental or compliance issues
│  ├─ Lease terms and renewal risks
│  └─ Geographic concentration
│
├─ Item 7: MD&A → Capital Allocation
│  ├─ Capital expenditures (historic, planned)
│  ├─ Asset depreciation/amortization
│  ├─ Working capital trends
│  ├─ Inventory levels and obsolescence
│  ├─ R&D spending as % of revenue
│  └─ Strategic facility investments
│
├─ Item 1A: Risk Factors → Operational Risks
│  ├─ "Single-source supplier dependency"
│  ├─ "Supply chain disruption in [region]"
│  ├─ "Manufacturing capacity constraints"
│  ├─ "Key facility located in [geopolitical risk area]"
│  ├─ "Technology infrastructure cybersecurity risks"
│  └─ "Just-in-time supply model vulnerability"
│
└─ Item 10/11: Executive Officers
   └─ Key person risk (CTO, COO, head of manufacturing)

RAPTOR EXTRACTION:
├─ Supply chain resilience: "We operate facilities in [countries]; 
│  X% of components from single source; lead times = Y days"
├─ Mitigation strategy: "We maintain Z-day safety stock and dual-source 
│  critical inputs; geographic diversification includes [locations]"
├─ Capacity signal: "Utilization rate = X%; planned capex = $Y; 
│  new capacity comes online [date]"
├─ Technology dependency: "Proprietary systems handle [function]; 
│  10% of revenue from [legacy product]; R&D = X% of revenue"
└─ Asset quality: "Facilities average age = X years; 
   depreciation rate = Y%; expected capex cycle = Z years"

SCORING RUBRIC:
┌──────────────────────────────────────────────────────────────┐
│ 9-10: | Diversified supply, high utilization, strong capex, │
│        | low tech risk, no single points of failure          │
│ 7-8:  | Adequate diversification, normal utilization,       │
│        | consistent R&D investment                           │
│ 5-6:  | Some concentration risk, capacity concerns,         │
│        | moderate tech/supplier dependency                   │
│ 3-4:  | High geographic or supplier concentration,          │
│        | capacity constraints, tech obsolescence risk        │
│ 0-2:  | Single-source dependency, facility disruption,      │
│        | imminent capex needs, critical systems at risk      │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (Manufacturing):
  Item 1: "We operate 4 facilities: 2 in USA, 1 in Mexico, 1 in Vietnam.
           78% of components sourced from 6 suppliers; 3 are single-source
           for critical items. We maintain 60-day safety stock."
  
  Item 2: "Facilities totaling 500K sq ft; 60% owned, 40% leased.
           Lease terms: 5-8 years with renewal options. 
           Facilities average age: 8 years."
  
  Item 7: "Capex = $12M (4% of revenue). Depreciation = $8M.
           Working capital increased $2M due to inventory buildup.
           R&D = $5M (2% of revenue)."
  
  Item 1A: "Vietnam geopolitical tensions could disrupt supply.
            We have no alternative source for [material] if [supplier] fails."
  
  → RAPTOR clusters:
    MACRO: "Geographic diversification with Vietnam concentration risk"
      └─ MICRO: 4 facilities across 3 countries; but Vietnam is critical
    MACRO: "Supply chain has bottlenecks but actively mitigated"
      └─ MICRO: 3 single-sources but 60-day safety stock and dual-sourcing plan
    MACRO: "Balanced capex and asset investment"
      └─ MICRO: $12M capex = 4% revenue; depreciation manageable
    SCORE: 6.8 (adequate but Vietnam risk and single-sources concerning)
```

---

### 4. MANAGEMENT EXECUTION Dimension
**Score: Has management delivered on promises?**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MANAGEMENT EXECUTION                              │
│     (Strategy Delivery, Guidance Accuracy, Leadership Alignment)     │
└─────────────────────────────────────────────────────────────────────┘

PRIMARY SOURCE SECTIONS:
├─ Item 1: Business → Strategy & Acquisition Criteria
│  ├─ Management's stated strategic priorities
│  ├─ Growth initiatives (organic, M&A)
│  ├─ Capital allocation philosophy
│  ├─ Target customer profiles and market segments
│  └─ Competitive positioning strategy
│
├─ Item 7: MD&A → Guidance Variance Analysis
│  ├─ Prior guidance vs. actual results
│  ├─ Explanations for variances
│  ├─ Management commentary on execution
│  ├─ Guidance changes (raises, cuts)
│  ├─ M&A integration status and realized synergies
│  └─ Strategic initiative status updates
│
├─ Item 10: Directors & Executive Officers
│  ├─ CEO/CFO tenure and track record
│  ├─ Prior company performance (where they worked)
│  ├─ Industry expertise and relevant experience
│  ├─ Board diversity and independence
│  └─ Committee composition (audit, compensation)
│
├─ Item 11: Executive Compensation
│  ├─ Metrics tied to executive bonuses (revenue, margin, etc.)
│  ├─ Actual bonus payouts vs. targets
│  ├─ Clawback provisions and enforcement
│  ├─ Long-term equity incentives (stock options, RSUs)
│  ├─ Compensation philosophy and peer benchmarking
│  └─ Executive turnover
│
├─ Item 1A: Risk Factors → Management Risks
│  ├─ "Dependence on key management personnel"
│  ├─ "Competition for talent"
│  ├─ "Management experience in [industry/model]"
│  └─ "Potential conflicts of interest with major shareholders"
│
└─ Item 2: Properties / Item 3: Legal
   └─ Any litigation involving executives (mismanagement, fraud, etc.)

RAPTOR EXTRACTION:
├─ Guidance track record: "FY2024 guided 20% revenue growth, achieved 18%
   (miss = -2%). FY2025 guide 15% growth; management attributes prior 
   miss to [macro/execution/market factors]"
├─ Strategy alignment: "We identified 3 strategic priorities in 2023:
   (1) Expand in [market]; (2) Reduce churn; (3) Improve margins.
   FY2024 results: (1) achieved, (2) achieved, (3) missed due to [reason]"
├─ M&A execution: "Acquired [company] for $X in 2023; synergies were
   initially $Y, now tracking to $Z (miss = $[diff]). Integration 
   on track for [date]"
├─ Incentive alignment: "CEO bonus is 50% revenue, 30% EBITDA margin,
   20% customer retention. FY2024: CEO received 75% of target bonus
   due to [miss]. Clawback: $[X] recovered."
└─ Leadership depth: "CEO has 15 years in [industry]; CFO spent 8 years
   at [peer company]. 3 of 5 board members are independent.
   No executive turnover last 2 years."

SCORING RUBRIC:
┌──────────────────────────────────────────────────────────────┐
│ 9-10: | Consistent guidance beats; clear strategy execution; │
│        | strong incentive alignment; stable leadership team   │
│ 7-8:  | Generally meets guidance; executing on priorities;   │
│        | reasonable compensation alignment                    │
│ 5-6:  | Mixed guidance track record; strategy unclear or     │
│        | partially executed; compensation concerns            │
│ 3-4:  | Repeated guidance misses; strategy pivots; high      │
│        | turnover; weak incentive alignment                   │
│ 0-2:  | Consistent guidance misses; lost credibility;        │
│        | failed M&A; key person departures; governance issues │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (Mature SaaS Company):
  Item 1: "Our strategy focuses on: (1) 30% organic growth through 
           product innovation, (2) 2-3 acquisitions/year for tuck-in 
           synergies, (3) expand into [new vertical]."
  
  Item 7: "FY2024 revenue $500M (+22% YoY; guided 25% miss = -3%).
           Organic growth 18%, M&A contributed 4%.
           We under-modeled [customer segment] churn.
           Revised FY2025 guide: 20% (down from prior 24% guide)."
  
  Item 10: "CEO (Sarah Chen, 12 yrs in SaaS) joined us 3 yrs ago from
            [competitor where she drove 2 successful M&As]. 
            CFO (John Smith, 7 yrs w/ us, CPA) prior controller at [firm].
            Board: 5 members; 4 independent; 1 shareholder rep."
  
  Item 11: "CEO target bonus = $2M (50% revenue growth, 30% gross margin,
            20% retention). FY2024 actual: $1.4M (70% of target due to
            revenue miss). Clawback policy in place; none triggered.
            CEO owns 0.5% stock (aligned with shareholders)."
  
  → RAPTOR clusters:
    MACRO: "Clear strategy with track record of partial execution"
      └─ MICRO: 22% growth vs 25% guidance; miss attributed to churn
    MACRO: "Management has relevant experience but trust eroding"
      └─ MICRO: CEO successful at [competitor]; 3 yrs here is mixed
    MACRO: "Incentives align but miss penalties limited"
      └─ MICRO: 70% bonus payout for -3% miss is lenient
    SCORE: 6.5 (strategy clear but execution slipping; watch trajectory)
```

---

### 5. REGULATORY & LICENSING Dimension
**Score: Are regulatory/legal risks manageable?**

```
┌─────────────────────────────────────────────────────────────────────┐
│              REGULATORY & LICENSING DIMENSION                       │
│     (Compliance, Pending Litigation, Regulatory Actions)            │
└─────────────────────────────────────────────────────────────────────┘

PRIMARY SOURCE SECTIONS:
├─ Item 1A: Risk Factors → Regulatory/Legal Subsection
│  ├─ "We are subject to [regulatory agency] oversight"
│  ├─ "New regulations in [jurisdiction] could impact [business]"
│  ├─ "We may not obtain/maintain [license/permit]"
│  ├─ "Antitrust concerns in [industry/deal]"
│  ├─ "Environmental compliance costs are [X]"
│  ├─ "Data privacy regulation (GDPR, CCPA) impacts [product]"
│  └─ "We have received [warning letters, enforcement actions]"
│
├─ Item 1: Business → Regulatory Context
│  ├─ Regulatory regime applicable to business model
│  ├─ Licensing dependencies (FDA, FCC, financial services, etc.)
│  ├─ Compliance cost structure
│  ├─ Impact of regulation on competitive position
│  └─ Lobbying/government relations efforts
│
├─ Item 3: Legal Proceedings (Most Important Section)
│  ├─ Material litigation (pending, settled, threatened)
│  ├─ Amount claimed / potential exposure
│  ├─ Outcome probability (management assessment)
│  ├─ Estimated resolution timeframe
│  ├─ Insurance coverage (if any)
│  ├─ Regulatory investigations or enforcement actions
│  ├─ Environmental liabilities or remediation obligations
│  └─ Settlement agreements or consent decrees
│
├─ Item 7: MD&A → Compliance Costs & Contingencies
│  ├─ Compliance/legal spending trends
│  ├─ Accrued liabilities for pending litigation
│  ├─ Insurance recoveries or subrogation
│  ├─ Remediation or settlement costs incurred
│  └─ Changes in regulatory landscape
│
└─ Item 1C: Cybersecurity (if applicable)
   ├─ Data breach history
   ├─ Cybersecurity incidents and response
   └─ Data privacy compliance

RAPTOR EXTRACTION:
├─ Regulatory exposure: "We operate under [FDA/EPA/FCC] oversight.
   Our primary product requires [approval type]. Timeline = [date].
   Failure to obtain approval would eliminate [X]% of revenue."
├─ Litigation exposure: "3 pending lawsuits totaling $50M claimed damage.
   Product liability suit filed [date]; settlement unlikely before [date].
   We maintain $100M insurance with $25M deductible per occurrence.
   Estimated exposure after insurance: $25M."
├─ Investigation risk: "We received [agency] subpoena in [date] regarding
   [subject]. We are cooperating. No indictment expected per management.
   Potential fine range: $[X] to $[Y]."
├─ Compliance cost: "Legal and compliance spending = $8M (2% of revenue);
   up 25% YoY due to [new regulation]. We expect [new initiative] will
   add $[X]M in recurring compliance cost."
└─ Mitigation actions: "We amended [policy/practice] to address [risk].
   We hired [expert] to oversee [compliance program]. Third-party audit
   confirmed no material gaps."

SCORING RUBRIC:
┌──────────────────────────────────────────────────────────────┐
│ 9-10: | No material litigation; no enforcement actions;      │
│        | regulatory compliant; stable regulatory environment  │
│ 7-8:  | Routine litigation (patent, employment); manageable  │
│        | fines; regulatory environment stable                 │
│ 5-6:  | Moderate litigation exposure ($10-50M); regulatory    │
│        | changes require strategic response                   │
│ 3-4:  | Material litigation ($50M+); enforcement action or    │
│        | investigation pending; major regulatory change       │
│ 0-2:  | Existential litigation risk; significant fines or     │
│        | sanctions pending; license/permit at risk            │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (Pharma Company):
  Item 1: "Our products are subject to FDA approval under [pathway].
           We hold [number] approvals. Maintenance requires [compliance].
           Any approval loss could eliminate [X]% of revenue."
  
  Item 1A: "FDA could impose new labeling requirements, restricting use.
            Clinical trials of our pipeline could fail regulatory review.
            Patent expiration of [drug] in 2027 exposes us to generics."
  
  Item 3: "Litigation:
           • Product liability lawsuit filed [date], claiming $20M damages
             for [injury]. Scheduled trial [date]. Insurance covers $10M.
           • Patent infringement counterclaim from [competitor] alleging
             we infringe 2 patents. Damages sought: $15M. Trial [date].
             Our counsel believes we have strong defense.
           • Wage and hour class action (n=[#] employees). Estimated
             settlement range: $2-5M. Resolution expected [date].
           Total estimated exposure: $12-20M after insurance."
  
  Item 7: "Accrued legal liabilities: $8M (vs $5M prior year).
           Legal spend: $3M (1.5% of revenue); up 30% YoY due to
           litigation. Insurance recoveries: $2M received."
  
  → RAPTOR clusters:
    MACRO: "Regulatory framework is stringent but manageable"
      └─ MICRO: FDA approval pathway established; patent cliff in 2027
    MACRO: "Material litigation exposure but partially insured"
      └─ MICRO: 3 cases totaling $12-20M exposure; $10M+ insured
    MACRO: "Patent risk significant but expected"
      └─ MICRO: Generic entry in 2027 is known risk; pipeline fills gap
    SCORE: 5.8 (moderate-high regulatory risk; material litigation;
            patent cliff manageable with pipeline)
```

---

## Summary: Section Prioritization by Extraction Quality

| Section | Quality | Effort | Signal |
|---------|---------|--------|--------|
| Item 1A (Risk Factors) | ⭐⭐⭐⭐⭐ | Medium | All 5 dimensions |
| Item 1 (Business) | ⭐⭐⭐⭐⭐ | Medium | Industry, Ops, Mgmt |
| Item 7 (MD&A) | ⭐⭐⭐⭐ | High | Revenue, Operations |
| Item 3 (Legal) | ⭐⭐⭐⭐ | Low | Regulatory only |
| Item 11 (Comp) | ⭐⭐⭐ | High | Mgmt Execution |
| Item 10 (Officers) | ⭐⭐⭐ | Low | Mgmt background |
| Item 2 (Properties) | ⭐⭐ | Low | Ops, Regulatory |
| Item 6 (Financial Data) | ⭐ | None | (Quantitative module) |

**Recommendation**: Extract Items 1, 1A, 3, 7 in full. Item 11 partially (focus on pay-for-performance metrics and clawbacks).

---

## HAVA Specific Mapping (SPAC Example)

Since HAVA is a **blank-check company** (SPAC), the 10-K has unusual characteristics:

```
HAVA Item 1: Business (pp. 1–6)
  § General description
    → Maps to INDUSTRY_BUSINESS: "We will target [criteria]"
    → Maps to MANAGEMENT_EXECUTION: "Our team background is [...]"
  
  § Background & Competitive Strengths
    → Maps to MANAGEMENT_EXECUTION: Track record of sponsors
  
  § Business Strategy & Acquisition Criteria
    → Maps to INDUSTRY_BUSINESS: Profile of target companies
    → Maps to MANAGEMENT_EXECUTION: Strategic priorities
  
  § Facilities & Employees (minimal for SPAC)
    → Maps to OPERATIONAL_ASSET: Trust Account structure (minimal)

HAVA Item 1A: Risk Factors (referenced to prospectus, limited in 10-K)
  § SPAC-specific risks
    → Maps to REGULATORY_LICENSING: Shareholder vote, redemption
    → Maps to REVENUE_STABILITY: Combination timeline uncertainty
    → Maps to MANAGEMENT_EXECUTION: Sponsor conflicts of interest

HAVA Item 7: MD&A (pp. 9–11)
  § Formation costs & operations
    → Maps to MANAGEMENT_EXECUTION: Cost burn rate
  § Trust Account composition
    → Maps to OPERATIONAL_ASSET: Capital sufficiency
  § Interest earned
    → Maps to REVENUE_STABILITY: "Revenue" during pre-deal period

HAVA Item 3: Legal Proceedings
  § Minimal (typical for new SPACs)
    → No material signal for first year

HAVA Item 11: Executive Compensation
  § Not yet filed (pre-deal SPAC has minimal comp)
    → Defer until post-combination
```

**Key Insight for SPACs**: Focus RAPTOR extraction on Item 1 (Management track record + acquisition criteria) and Item 1A/7 (combination timeline risk + capital sufficiency). Revenue Stability is less relevant pre-deal.

---

## Next Step: Custom Prompting

Once you've extracted sections using RAPTOR, you can refine the scoring prompts by dimension:

```python
DIMENSION_PROMPTS = {
    'INDUSTRY_BUSINESS': """
You are evaluating [COMPANY]'s competitive position in [INDUSTRY].

From the extracted 10-K narrative, assess:
1. Market share or competitive positioning (specific numbers if available)
2. Competitive advantages or differentiation
3. Industry cyclicality or long-term trends
4. Supplier/customer concentration risks
5. Threat from disruption or new competitors

Score on 0–10 scale. Provide specific evidence from the text.
""",
    'REVENUE_STABILITY': """
You are evaluating revenue quality and predictability for [COMPANY].

From the extracted 10-K narrative, assess:
1. % of revenue that is recurring vs. non-recurring
2. Customer concentration (top 3, top 5, top 10 as % of revenue)
3. Customer churn or renewal rates
4. Contract length and backlog/pipeline visibility
5. Significant customer wins or losses and reasons

Score on 0–10 scale. Provide specific metrics and source citations.
""",
    # ... (similar for other dimensions)
}
```

This ensures each dimension gets the **most relevant context** from the extracted sections.

