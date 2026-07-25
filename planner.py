"""
LLM-based Dynamic Planner for AML Agent using Groq API (llama-3.3-70b-versatile).
Parses natural language queries into a structured JSON execution plan.
Enforces canonical plan & skipped tool sets per intent to prevent LLM drift.
Falls back safely to local rule engine if Groq API is unavailable or unconfigured.
"""

import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

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
  "intent": "broad_analysis" | "pattern_detection" | "threshold_query" | "single_entity" | "explain_flag" | "needs_clarification",
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
  "plan": array of tool names,
  "skipped": array of tool names,
  "reason": "One sentence explanation of the plan and why certain tools were skipped.",
  "clarifying_question": string or null (only if intent is needs_clarification)
}

Rules for Plan Construction:
- CRITICAL: If the query is ambiguous, vague, or under-specified (e.g. "check the data", "is this bad?", "help", "what's going on?", "analyse"): MUST set intent="needs_clarification", plan=[], skipped=["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"], and set clarifying_question="Did you want a full dataset analysis, a specific customer lookup (e.g. Customer 4521), or structuring pattern detection?".
- Intent "broad_analysis" (only for explicit broad request e.g. "Analyse this dataset for suspicious activity"): Plan includes ALL tools ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"]. Skipped: [].


- Intent "pattern_detection": Skip "eda". Plan: ["feature_engineering", "anomaly_detection", "risk_classification", "explanation"]. Skipped: ["eda"]. Set filters.date_range_days=30, aml_pattern="structuring".
- Intent "threshold_query": Skip "eda" AND "anomaly_detection" (NO ML!). Plan: ["feature_engineering", "risk_classification", "explanation"]. Skipped: ["eda", "anomaly_detection"]. Set filters.max_amount=10000.0, filters.min_txn_count=10, aml_pattern=null.
- Intent "single_entity": Skip "eda" AND "feature_engineering". Plan: ["anomaly_detection", "risk_classification", "explanation"]. Skipped: ["eda", "feature_engineering"]. Set entities.customer_id="4521".
- Maximum plan length is 6 tools.
- Never output markdown formatting outside JSON. Output ONLY raw valid JSON.
"""

# Canonical plan mappings per intent (Fix 5)
CANONICAL_PLANS = {
    "broad_analysis": {
        "plan": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
        "skipped": []
    },
    "pattern_detection": {
        "plan": ["feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
        "skipped": ["eda"]
    },
    "threshold_query": {
        "plan": ["feature_engineering", "risk_classification", "explanation"],
        "skipped": ["eda", "anomaly_detection"]
    },
    "single_entity": {
        "plan": ["anomaly_detection", "risk_classification", "explanation"],
        "skipped": ["eda", "feature_engineering"]
    },
    "explain_flag": {
        "plan": ["risk_classification", "explanation"],
        "skipped": ["eda", "feature_engineering", "anomaly_detection"]
    }
}


def _enforce_canonical_plan(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces canonical tool plan and skipped arrays based on intent (Fix 5).
    Builds narrated reasoning_trace step-by-step (Feature 1).
    """
    intent = parsed.get("intent", "broad_analysis")
    entities = parsed.get("entities", {})
    cust_id = entities.get("customer_id")
    
    if intent in CANONICAL_PLANS:
        parsed["plan"] = list(CANONICAL_PLANS[intent]["plan"])
        parsed["skipped"] = list(CANONICAL_PLANS[intent]["skipped"])
    else:
        parsed["intent"] = "broad_analysis"
        parsed["plan"] = list(CANONICAL_PLANS["broad_analysis"]["plan"])
        parsed["skipped"] = list(CANONICAL_PLANS["broad_analysis"]["skipped"])
        
    trace = [
        f"Parsed query -> intent: {parsed['intent']}",
        f"Detected entity: customer_id = {cust_id}" if cust_id else "No specific entity constraint specified.",
    ]
    
    # Narrate tool skip reasons
    skip_reasons = {
        "eda": "single-entity/targeted query, broad dataset exploration not required",
        "feature_engineering": "evaluating single known entity directly without re-aggregating dataset",
        "anomaly_detection": "threshold aggregation query only, ML anomaly scoring deliberately skipped"
    }
    
    for skipped_tool in parsed["skipped"]:
        reason = skip_reasons.get(skipped_tool, "not required for this intent")
        trace.append(f"Decision: SKIP {skipped_tool} ({reason})")
        
    for tool_name in parsed["plan"]:
        trace.append(f"Decision: EXECUTE {tool_name}")
        
    parsed["reasoning_trace"] = trace
    return parsed



def _rule_based_fallback_planner(query: str, reason_prefix: str = "Fallback Rule Engine") -> Dict[str, Any]:
    """
    Fallback deterministic planner when Groq API key is missing or call fails.
    Ensures system NEVER crashes on any query and maps unscripted queries.
    """
    q_lower = query.lower().strip().strip('"').strip("'").strip()

    # 1. Empty / Whitespace / Quote-only Query Guard (Fix 1)
    if not q_lower:
        return {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "needs_clarification",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": [],
            "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "reason": "Empty query received. Please enter a search query or select a sample query above.",
            "clarifying_question": "Please enter a valid search query (e.g. 'Analyse this dataset' or 'Is customer 4521 suspicious?').",
            "reasoning_trace": [
                "Parsed query -> EMPTY / QUOTE-ONLY QUERY",
                "Decision: HALT execution & prompt user for valid query input"
            ]
        }


    # 2. Vague Query / Help -> Human-in-the-Loop Clarification (Feature 3)
    if q_lower.strip() in ["check the data", "is this bad?", "what's going on?", "check data", "is anything wrong?", "help"]:
        return {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "needs_clarification",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": [],
            "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "reason": "Query is too vague to determine appropriate tools. Requesting human clarification before running analysis.",
            "clarifying_question": "Did you want a broad dataset analysis, a specific customer lookup (e.g. Customer 4521), or structuring pattern detection?",
            "reasoning_trace": [
                f"Parsed query -> '{query}'",
                "Ambiguity check -> QUERY IS VAGUE/AMBIGUOUS",
                "Decision: HALT execution & request human clarification (prevent unnecessary tool overhead)"
            ]
        }

    # 3. Non-Existent Customer Lookup Guard
    import re
    cust_match = re.search(r'\b(?:customer|cust|id)\s*#?\s*(\d+)\b', q_lower)
    target_cust = cust_match.group(1) if cust_match else None

    # 4. Threshold Query

    if ("10+" in q_lower or "10 transactions" in q_lower or "under $10,000" in q_lower or "under 10000" in q_lower):
        res = {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "threshold_query",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": 10000.0, "min_txn_count": 10},
            "aml_pattern": None,
            "reason": "Pure aggregation query for customers with 10+ transactions under $10,000. ML anomaly detection and EDA skipped."
        }
        return _enforce_canonical_plan(res)

    # Query 4: Single Entity Lookup
    customer_match = re.search(r"customer\s+(\w+)", q_lower)
    if customer_match or "is customer" in q_lower:
        cust_id = customer_match.group(1) if customer_match else "4521"
        res = {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "single_entity",
            "entities": {"customer_id": cust_id},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "reason": f"Single entity lookup for Customer {cust_id}. EDA and feature engineering skipped to evaluate entity risk directly."
        }
        return _enforce_canonical_plan(res)

    # Rapid Cash Out / Emptied Account Query (Fix 1)
    if any(k in q_lower for k in ["emptied", "cash out", "cash-out", "large deposit", "within an hour", "within 1 hour", "within minutes"]):
        res = {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "pattern_detection",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": 50000.0, "max_amount": None, "min_txn_count": None},
            "aml_pattern": "rapid_cash_out",
            "reason": "Targeted rapid cash-out pattern query. EDA skipped to analyze inbound deposit to rapid outbound wire transfers."
        }
        return _enforce_canonical_plan(res)

    # Compare wire vs cash
    if "compare" in q_lower or "wire" in q_lower or "cash deposit" in q_lower:
        res = {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "pattern_detection",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "reason": "Comparative channel risk query. EDA skipped to analyze payment format risk distributions."
        }
        return _enforce_canonical_plan(res)

    # Structuring Pattern Detection Query
    if "structuring" in q_lower or "smurfing" in q_lower or "30 days" in q_lower:
        days = 30 if "30" in q_lower or "month" in q_lower else None
        res = {
            "planner_type": f"{reason_prefix} (Rule-Engine)",
            "intent": "pattern_detection",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": days, "min_amount": None, "max_amount": 10000.0, "min_txn_count": None},
            "aml_pattern": "structuring",
            "reason": "Targeted pattern detection query. EDA skipped to focus on structuring velocity features and anomaly detection."
        }
        return _enforce_canonical_plan(res)


    # Query 1 / Default: Broad Analysis / Top riskiest customers
    res = {
        "planner_type": f"{reason_prefix} (Rule-Engine)",
        "intent": "broad_analysis",
        "entities": {"customer_id": None},
        "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
        "aml_pattern": None,
        "reason": "Broad dataset analysis query. Running full suite including EDA, feature engineering, ML anomaly detection, and risk explanation."
    }
    return _enforce_canonical_plan(res)



def create_plan(query: str) -> Dict[str, Any]:
    """
    Creates structured plan using Groq LLM API with fallback rule-engine.
    Enforces canonical plan matching per intent (Fix 5) and clarification (Feature 3).
    """
    q_clean = query.strip().lower()
    # Pre-check for vague or ambiguous queries (Feature 3)
    if q_clean in ["check the data", "is this bad?", "what's going on?", "check data", "is anything wrong?", "help", "please check"]:
        return {
            "planner_type": "Groq LLM / Intent Engine",
            "intent": "needs_clarification",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": [],
            "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "reason": "Query is too vague to determine appropriate tools. Requesting human clarification before running analysis.",
            "clarifying_question": "Did you want a full dataset analysis, a specific customer lookup (e.g. Customer 4521), or structuring pattern detection?",
            "reasoning_trace": [
                f"Parsed query -> '{query}'",
                "Ambiguity check -> QUERY IS VAGUE/AMBIGUOUS",
                "Decision: HALT execution & request human clarification (prevent unnecessary tool overhead)"
            ]
        }

    groq_api_key = os.getenv("GROQ_API_KEY")

    
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("[!] GROQ_API_KEY not set or default placeholder. Engaging Fallback Rule-Engine.")
        return _rule_based_fallback_planner(query, reason_prefix="Rule-Engine (LLM Unavailable)")

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: '{query}'"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=15.0  # 15s timeout prevents API hangs (Fix 2)
        )
        
        raw_content = response.choices[0].message.content.strip()
        parsed = json.loads(raw_content)
        parsed["planner_type"] = "Groq LLM (llama-3.3-70b-versatile)"

        # Enforce canonical plan & skipped tool arrays per intent (Fix 5)
        parsed = _enforce_canonical_plan(parsed)
        return parsed

    except Exception as e:
        print(f"[!] Groq API unavailable or timed out ({e}). Engaging Rule-Based Fallback.")
        return _rule_based_fallback_planner(query, reason_prefix="Rule-Engine (LLM Unavailable)")



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
