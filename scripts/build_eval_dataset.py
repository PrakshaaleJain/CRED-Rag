import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_file = data_dir / 'raptor_eval_benchmark.json'
    
    benchmark_data = [
        # --- ZETA (Zeta Global) ---
        # Factual
        {
            "id": "Q_FACTUAL_ZETA_001",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What percentage of Zeta's revenue in the year ended December 31, 2025, was derived from super-scaled customers?",
            "ground_truth": "87% of Zeta's revenue for the year ended December 31, 2025, was derived from super-scaled customers.",
            "expected_keywords": ["87%", "super-scaled"]
        },
        {
            "id": "Q_FACTUAL_ZETA_002",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What is the trailing-12-month revenue threshold for a customer to be defined as 'super-scaled' by Zeta?",
            "ground_truth": "Zeta defines super-scaled customers as those from which they have generated trailing-12-month revenue of at least $1,000,000.",
            "expected_keywords": ["$1,000,000", "1,000,000", "one million"]
        },
        {
            "id": "Q_FACTUAL_ZETA_003",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "How many super-scaled customers did Zeta have as of December 31, 2025?",
            "ground_truth": "Zeta had 184 super-scaled customers as of December 31, 2025.",
            "expected_keywords": ["184"]
        },
        {
            "id": "Q_FACTUAL_ZETA_004",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What percentage of Zeta's revenues were contributed by the consumer & retail industry vertical for the year ended December 31, 2025?",
            "ground_truth": "The consumer & retail industry vertical contributed 24% of Zeta's revenues for the year ended December 31, 2025.",
            "expected_keywords": ["24%"]
        },
        {
            "id": "Q_FACTUAL_ZETA_005",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "Did Zeta's top ten customers account for more than one-third of total revenue in 2025?",
            "ground_truth": "Yes, in 2025, Zeta's top ten customers accounted for more than one-third of total revenue.",
            "expected_keywords": ["top ten", "one-third", "Yes"]
        },
        
        # Conceptual
        {
            "id": "Q_CONCEPTUAL_ZETA_001",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "How does the political cycle affect Zeta's quarterly and annual operating results?",
            "ground_truth": "Zeta generally experiences higher revenues during congressional election years, and particularly during presidential election years and, to a lesser extent, midterm election years. This cyclicality may affect the comparability of results between non-election years.",
            "expected_keywords": ["political", "election", "presidential", "midterm", "higher revenues"]
        },
        {
            "id": "Q_CONCEPTUAL_ZETA_002",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What factors make Zeta's usage-based pricing model less stable than subscription-based pricing?",
            "ground_truth": "Customers do not have automatic renewal or exclusive obligations, and they can decrease their overall marketing spend or cease usage for any reason (like insufficient returns). Because a substantial portion of revenue is usage-based, a reduction in usage by super-scaled customers significantly impacts revenue.",
            "expected_keywords": ["usage-based", "decrease", "marketing spend", "no automatic renewal"]
        },
        {
            "id": "Q_CONCEPTUAL_ZETA_003",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What are the primary challenges Zeta faces regarding its sales cycle for enterprise customers?",
            "ground_truth": "Zeta often has long sales cycles requiring considerable time and expense to evaluate organizational needs and educate potential customers. Decisions are based on factors beyond platform features, such as capital budgets and economic uncertainty, meaning substantial prospecting resources may not result in revenue.",
            "expected_keywords": ["long sales cycle", "time", "expense", "educate"]
        },
        {
            "id": "Q_CONCEPTUAL_ZETA_004",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "How does seasonality impact Zeta's liquidity and revenue from quarter to quarter?",
            "ground_truth": "Marketing activity is historically higher in the fourth quarter to coincide with the holiday shopping season. Consequently, the subsequent first quarter tends to reflect lower activity levels and lower performance.",
            "expected_keywords": ["fourth quarter", "holiday", "first quarter", "lower activity"]
        },
        {
            "id": "Q_CONCEPTUAL_ZETA_005",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What compliance risks does Zeta face regarding data protection in Europe?",
            "ground_truth": "Zeta is subject to the EU GDPR and UK GDPR, facing strict requirements for processing personal data. Noncompliance can result in regulatory enforcement and fines up to €20 million or 4% of annual global revenues under each regime independently.",
            "expected_keywords": ["GDPR", "fines", "€20 million", "4%"]
        },
        
        # Strategic
        {
            "id": "Q_STRATEGIC_ZETA_001",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What is Zeta's long-term growth strategy regarding its customer base?",
            "ground_truth": "Zeta's strategy focuses on adding, growing, and retaining super-scaled customers. They intend to continually win new super-scaled customers, educate existing ones to increase platform usage, and capture a larger share of their marketing spend. They are abandoning metrics for regular 'scaled' customers to focus entirely on super-scaled growth.",
            "expected_keywords": ["super-scaled", "retain", "grow", "marketing spend"]
        },
        {
            "id": "Q_STRATEGIC_ZETA_002",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "How does Zeta plan to navigate the intensely competitive marketing technology industry against larger, consolidated competitors?",
            "ground_truth": "Zeta plans to continuously respond to evolving trends by investing in their platform to maintain technological competitiveness, enhancing current products, and developing new solutions to meet customer demands, avoiding commercial obsolescence.",
            "expected_keywords": ["invest", "technological competitiveness", "enhance", "develop new"]
        },
        {
            "id": "Q_STRATEGIC_ZETA_003",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What are Zeta's strategic concerns regarding cybersecurity and the integration of AI?",
            "ground_truth": "Zeta acknowledges that integrating AI poses new and unknown cybersecurity risks. Threat actors change techniques frequently. Zeta invests in security measures but warns that breaches due to employee error, remote work vulnerabilities, or third-party extortion could disrupt operations and that insurance may not cover all liabilities.",
            "expected_keywords": ["AI", "cybersecurity", "unknown risks", "employee error", "remote work"]
        },
        {
            "id": "Q_STRATEGIC_ZETA_004",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "How is Zeta addressing the regulatory landscape surrounding AI algorithms and data bias?",
            "ground_truth": "Zeta faces evolving regulations (like the FTC Act, Equal Credit Opportunity Act, and EU AI Act) regarding AI bias and antidiscrimination. Their strategy requires adapting processes to ensure algorithms do not unfairly discriminate, which increases compliance costs and may restrict data collection needed to train AI.",
            "expected_keywords": ["bias", "antidiscrimination", "EU AI Act", "compliance costs"]
        },
        {
            "id": "Q_STRATEGIC_ZETA_005",
            "company_id": "ZETA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What is management's view on their reliance on third-party data suppliers for their platform's success?",
            "ground_truth": "Zeta views maintaining relationships with third-party data suppliers as critical. If suppliers withdraw data or if ties are terminated for commercial or regulatory reasons, Zeta's ability to provide products would be materially adversely impacted, and management warns there is no assurance alternative sources can be found on acceptable terms.",
            "expected_keywords": ["third-party data", "suppliers", "withdraw", "alternative sources"]
        },
        
        # --- HAVA ---
        # Factual
        {
            "id": "Q_FACTUAL_HAVA_001",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What is the CIK identifier for HAVA as listed in the filing?",
            "ground_truth": "The CIK identifier for HAVA in the document is 0002042460.",
            "expected_keywords": ["0002042460"]
        },
        {
            "id": "Q_FACTUAL_HAVA_002",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "For what fiscal year ending date is the HAVA report filed?",
            "ground_truth": "The HAVA report is filed for the fiscal year ended December 31, 2025.",
            "expected_keywords": ["December 31, 2025", "2025"]
        },
        {
            "id": "Q_FACTUAL_HAVA_003",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What accounting standard is used for HAVA's Fair Value Inputs Level 1?",
            "ground_truth": "HAVA uses the US-GAAP accounting standard for Fair Value Inputs Level 1 (us-gaap:FairValueInputsLevel1Member).",
            "expected_keywords": ["US-GAAP", "FairValueInputsLevel1"]
        },
        {
            "id": "Q_FACTUAL_HAVA_004",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What specific measurement input does HAVA list regarding DeSPAC probability?",
            "ground_truth": "HAVA lists 'MeasurementInputProbabilityOfDeSPACMember'.",
            "expected_keywords": ["MeasurementInputProbabilityOfDeSPACMember", "ProbabilityOfDeSPAC"]
        },
        {
            "id": "Q_FACTUAL_HAVA_005",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Factual",
            "query": "What measurement input does HAVA record regarding marketability?",
            "ground_truth": "HAVA records 'MeasurementInputDiscountForLackOfMarketabilityMember'.",
            "expected_keywords": ["DiscountForLackOfMarketability", "marketability"]
        },
        
        # Conceptual
        {
            "id": "Q_CONCEPTUAL_HAVA_001",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "How is the probability of a DeSPAC transaction treated in HAVA's fair value measurements?",
            "ground_truth": "It is treated as a specific measurement input variable (MeasurementInputProbabilityOfDeSPACMember) used in calculating fair value.",
            "expected_keywords": ["measurement input", "fair value"]
        },
        {
            "id": "Q_CONCEPTUAL_HAVA_002",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What does the inclusion of a discount for lack of marketability indicate about HAVA's assets?",
            "ground_truth": "It indicates that some of HAVA's measured assets or equity instruments are illiquid or cannot be easily converted to cash on public markets, necessitating a valuation discount.",
            "expected_keywords": ["discount", "illiquid", "marketability"]
        },
        {
            "id": "Q_CONCEPTUAL_HAVA_003",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "Why would HAVA use US-GAAP Level 1 inputs for some fair value measurements?",
            "ground_truth": "Level 1 inputs refer to quoted prices in active markets for identical assets, meaning HAVA holds some highly liquid, actively traded financial instruments.",
            "expected_keywords": ["Level 1", "active markets", "quoted prices"]
        },
        {
            "id": "Q_CONCEPTUAL_HAVA_004",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What role does the Chief Financial Officer (CFO) play in the XBRL tagging context for HAVA?",
            "ground_truth": "The CFO is tagged as a specific member (ChiefFinancialOfficerMember) in the XBRL taxonomy, likely signing off on or taking responsibility for the disclosed financial measurements.",
            "expected_keywords": ["XBRL", "ChiefFinancialOfficerMember"]
        },
        {
            "id": "Q_CONCEPTUAL_HAVA_005",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Conceptual",
            "query": "What is the significance of the Conversion Price measurement input?",
            "ground_truth": "The Conversion Price input (MeasurementInputConversionPriceMember) is used to value convertible securities, indicating HAVA has debt or preferred stock that can convert to common equity.",
            "expected_keywords": ["convertible", "ConversionPrice", "equity"]
        },
        
        # Strategic
        {
            "id": "Q_STRATEGIC_HAVA_001",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "Based on the measurement inputs, what is HAVA's primary corporate structural strategy?",
            "ground_truth": "HAVA's inclusion of 'MeasurementInputProbabilityOfDeSPACMember' strongly indicates it is a Special Purpose Acquisition Company (SPAC) whose primary strategic goal is executing a business combination (DeSPAC).",
            "expected_keywords": ["SPAC", "DeSPAC", "business combination"]
        },
        {
            "id": "Q_STRATEGIC_HAVA_002",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "How does HAVA's valuation strategy manage the risk of illiquid securities?",
            "ground_truth": "HAVA systematically applies a Discount for Lack of Marketability to accurately reflect the reduced value and liquidity risk associated with untradeable assets prior to a potential DeSPAC event.",
            "expected_keywords": ["liquidity risk", "discount for lack of marketability", "untradeable"]
        },
        {
            "id": "Q_STRATEGIC_HAVA_003",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What capital structuring strategy is implied by HAVA's measurement inputs?",
            "ground_truth": "HAVA utilizes complex financial instruments, including those with conversion features (MeasurementInputConversionPriceMember), indicating a strategy of issuing convertible debt or warrants to attract investors.",
            "expected_keywords": ["convertible", "warrants", "capital structuring"]
        },
        {
            "id": "Q_STRATEGIC_HAVA_004",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What reporting strategy does HAVA use for asset transparency?",
            "ground_truth": "HAVA categorizes its assets using US-GAAP fair value hierarchies, clearly distinguishing between highly liquid Level 1 assets and more subjective internal valuations (DeSPAC probability) to maintain regulatory compliance and transparency.",
            "expected_keywords": ["US-GAAP", "fair value hierarchy", "transparency", "compliance"]
        },
        {
            "id": "Q_STRATEGIC_HAVA_005",
            "company_id": "HAVA",
            "fiscal_year": "2026",
            "abstraction_level": "Strategic",
            "query": "What is the ultimate risk factor driving HAVA's financial disclosures?",
            "ground_truth": "The ultimate risk factor is the successful execution of a DeSPAC transaction, as the probability of this event fundamentally underpins the valuation of their issued instruments and warrants.",
            "expected_keywords": ["execution", "DeSPAC", "valuation", "probability"]
        }
    ]
    
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=4)
        
    logging.info(f"Successfully generated 30 test cases in {benchmark_file}")

if __name__ == "__main__":
    main()
