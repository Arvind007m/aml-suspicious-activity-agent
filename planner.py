"""
LLM-based Dynamic Planner for AML Agent using Groq API (llama-3.3-70b-versatile).
Parses natural language queries into a structured JSON execution plan.
Falls back safely to local rule engine if Groq API is unavailable or unconfigured.
"""

import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from registry import VALID_TOOLS, validate_plan

SYSTEM_PROMPT = """You are an expert AI Planner for an Anti-Money-Laundering (AML) Detection System.
Your job is to analyze a natural-language query and generate a structured JSON execution plan.

You have access to 5 modular tools:
1. "eda": Exploratory data profiling, dataset overview, missing values, summary statistics.
2. "feature_engineering": Velocity, transaction frequency, rolling 24h sums, structuring/smurfing counts ($9k-$10k).
3. "anomaly_detection": ML IsolationForest scoring for multi-dimensional anomaly detection.
4. "risk_classification": Categorizes risk into Low, Medium, High based on aggregated risk scores.
5. "explanation": Generates plain-English reasons for flags and recommended SAR/Review escalation actions.

Your JSON output MUST follow this schema strictly:
{
  "intent": "broad_analysis" | "pattern_detection" | "threshold_query" | "single_entity" | "explain_flag",
  "entities": {
    "customer_id": string or null
  },
  "filters": {
    "date_range_days": integer or null,
    "min_amount": float or null,
    "max_amount": float or null,
    "min_txn_count": integer or null
  },
  "aml_pattern": "structuring" | "rapid_cash_out" | "amount_spike" | null,
  "plan": array of tool names from ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
  "skipped": array of tool names from ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
  "reason": "One sentence explanation of the plan and why certain tools were skipped."
}

Rules for Plan Construction:
- Intent "broad_analysis" (e.g. "Analyse dataset"): Include ALL tools ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"]. Skipped: [].
- Intent "pattern_detection" (e.g. "Find structuring patterns in last 30 days"): Skip "eda". Plan: ["feature_engineering", "anomaly_detection", "risk_classification", "explanation"]. Skipped: ["eda"]. Set filters.date_range_days=30, aml_pattern="structuring".
- Intent "threshold_query" (e.g. "Which customers made 10+ transactions under $10,000?"): Skip "eda" AND "anomaly_detection" (NO ML!). Plan: ["feature_engineering", "risk_classification", "explanation"]. Skipped: ["eda", "anomaly_detection"]. Set filters.max_amount=10000.0, filters.min_txn_count=10.
- Intent "single_entity" (e.g. "Is customer 4521 suspicious?"): Skip "eda" AND "feature_engineering". Plan: ["anomaly_detection", "risk_classification", "explanation"]. Skipped: ["eda", "feature_engineering"]. Set entities.customer_id="4521".
- Maximum plan length is 6 tools.
- Never output markdown formatting outside JSON. Output ONLY raw valid JSON.
"""


def _rule_based_fallback_planner(query: str, reason_prefix: str = "Fallback Rule Engine") -> Dict[str, Any]:
    """
    Fallback deterministic planner when Groq API key is missing or call fails.
    Ensures system NEVER crashes on any query.
    """
    q_lower = query.lower()

    # Query 3: Threshold Query (Check this BEFORE general customer word check)
    if ("10+" in q_lower or "10 transactions" in q_lower or "under $10,000" in q_lower or "under 10000" in q_lower):
        return {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "threshold_query",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": 10000.0, "min_txn_count": 10},
            "aml_pattern": "structuring",
            "plan": ["feature_engineering", "risk_classification", "explanation"],
            "skipped": ["eda", "anomaly_detection"],
            "reason": "Pure aggregation query for customers with 10+ transactions under $10,000. ML anomaly detection and EDA skipped."
        }

    # Query 4: Single Entity Lookup (Check for specific customer ID pattern)
    customer_match = re.search(r"customer\s+(\w+)", q_lower)
    if customer_match or "is customer" in q_lower:
        cust_id = customer_match.group(1) if customer_match else "4521"
        return {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "single_entity",
            "entities": {"customer_id": cust_id},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": ["anomaly_detection", "risk_classification", "explanation"],
            "skipped": ["eda", "feature_engineering"],
            "reason": f"Single entity lookup for Customer {cust_id}. EDA and feature engineering skipped to evaluate entity risk directly."
        }

    # Query 2: Pattern Detection (e.g. structuring)
    if "structuring" in q_lower or "pattern" in q_lower or "smurfing" in q_lower or "30 days" in q_lower:
        days = 30 if "30" in q_lower or "month" in q_lower else None
        return {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "pattern_detection",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": days, "min_amount": None, "max_amount": 10000.0, "min_txn_count": None},
            "aml_pattern": "structuring",
            "plan": ["feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "skipped": ["eda"],
            "reason": "Targeted pattern detection query. EDA skipped to focus on structuring velocity features and anomaly detection."
        }

    # Query 1 / Default: Broad Analysis
    return {
        "planner_type": f"{reason_prefix} (Rule-Engine)",
        "intent": "broad_analysis",
        "entities": {"customer_id": None},
        "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
        "aml_pattern": None,
        "plan": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
        "skipped": [],
        "reason": "Broad dataset analysis query. Running full suite including EDA, feature engineering, ML anomaly detection, and risk explanation."
    }



def create_plan(query: str) -> Dict[str, Any]:
    """
    Creates structured plan using Groq LLM API with fallback rule-engine.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("[!] GROQ_API_KEY not found in environment. Using Fallback Rule-Engine.")
        return _rule_based_fallback_planner(query, reason_prefix="Offline Fallback")

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: \"{query}\""}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)
        
        # Validate tool names in plan and skipped
        validated_plan = validate_plan(parsed.get("plan", []))
        validated_skipped = [t for t in parsed.get("skipped", []) if t in VALID_TOOLS]
        
        parsed["planner_type"] = "Groq LLM (llama-3.3-70b-versatile)"
        parsed["plan"] = validated_plan
        parsed["skipped"] = validated_skipped
        return parsed

    except Exception as e:
        print(f"[!] Groq API call failed or parse error: {e}. Falling back to Rule Engine.")
        return _rule_based_fallback_planner(query, reason_prefix="Groq Error Fallback")


if __name__ == "__main__":
    test_queries = [
        "Analyse this dataset for suspicious activity",
        "Find structuring patterns in the last 30 days",
        "Which customers made 10+ transactions under $10,000?",
        "Is customer 4521 suspicious?"
    ]
    
    print("==================================================")
    print("          TESTING LLM DYNAMIC PLANNER             ")
    print("==================================================\n")
    for q in test_queries:
        plan_res = create_plan(q)
        print(f"QUERY: \"{q}\"")
        print(f"PLANNER TYPE: {plan_res['planner_type']}")
        print(f"INTENT:       {plan_res['intent']}")
        print(f"PLAN:         {plan_res['plan']}")
        print(f"SKIPPED:      {plan_res['skipped']}")
        print(f"REASON:       {plan_res['reason']}\n" + "-"*50 + "\n")
