"""
Explanation Tool: Generates plain-English explanations and escalation recommendations.
Uses BATCHED LLM prompts (or high-performance rule templates) to stay well under rate limits.
"""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from registry import register_tool

load_dotenv()


def _generate_template_explanations(top_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    High-performance rule-template narrative generator (fallback / zero API calls).
    """
    explained = []
    for item in top_entities:
        cust_id = item.get("customer_id")
        risk_level = item.get("risk_level", "Low")
        struct_cnt = item.get("structuring_count", 0)
        cashout = item.get("rapid_cashout_flag", 0)
        velocity = item.get("velocity_24h", 0)
        anomaly_score = item.get("anomaly_score", 0.0)
        txn_cnt = item.get("txn_count_total", 0)
        total_usd = item.get("total_amount_usd", 0.0)

        reasons = []
        if struct_cnt >= 10:
            reasons.append(f"{struct_cnt} transactions systematically structured under $10,000 reporting threshold (Smurfing pattern)")
        elif struct_cnt >= 3:
            reasons.append(f"{struct_cnt} transactions structured near $10,000 reporting threshold")
            
        if cashout == 1:
            reasons.append("Rapid cash-out pattern detected: large inbound deposit followed by rapid outbound wire splits")
            
        if velocity >= 10:
            reasons.append(f"High transaction velocity ({velocity} txns in rolling 24-hour window)")
            
        if anomaly_score > 0.6:
            reasons.append(f"High machine-learning anomaly score ({anomaly_score:.2f}) from IsolationForest model")
            
        if not reasons:
            reasons.append(f"Total transaction volume ${total_usd:,.2f} across {txn_cnt} transactions within expected threshold boundaries")

        explanation_text = f"Customer {cust_id}: " + "; ".join(reasons) + "."

        # Escalation action mapping
        if risk_level == "High":
            escalation = "File SAR (Suspicious Activity Report) & Freeze Account"
        elif risk_level == "Medium":
            escalation = "Enhanced Due Diligence (EDD) Review"
        else:
            escalation = "Routine Monitoring"

        explained_item = dict(item)
        explained_item["explanation"] = explanation_text
        explained_item["escalation_action"] = escalation
        explained.append(explained_item)

    return explained


@register_tool("explanation")
def run_explanation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates explanations for top flagged items in a SINGLE batched LLM call or template fallback.
    """
    top_entities = context.get("top_suspicious_entities", [])
    if not top_entities:
        context["explanations"] = []
        return context

    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # If no key or offline mode, use template generator
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        context["explanations"] = _generate_template_explanations(top_entities)
        return context

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        prompt_entities = [
            {
                "customer_id": e.get("customer_id"),
                "risk_score": e.get("risk_score"),
                "risk_level": e.get("risk_level"),
                "structuring_count": e.get("structuring_count"),
                "velocity_24h": e.get("velocity_24h"),
                "rapid_cashout": e.get("rapid_cashout_flag"),
                "anomaly_score": e.get("anomaly_score")
            }
            for e in top_entities
        ]
        
        system_msg = """You are a senior AML compliance officer.
Provide concise 1-2 sentence plain-English explanations for each flagged customer based on their metrics.
Return a JSON object with key "explanations" which is an array of objects:
[
  {
    "customer_id": "4521",
    "explanation": "Customer executed 15 cash deposits under $10,000 within 24h, exhibiting classic smurfing structuring.",
    "escalation_action": "File SAR & Freeze Account"
  }
]
Valid escalation actions: "File SAR & Freeze Account", "Enhanced Due Diligence (EDD) Review", "Routine Monitoring".
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Flagged Customers Data: {json.dumps(prompt_entities)}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        raw_res = response.choices[0].message.content.strip()
        parsed = json.loads(raw_res)
        llm_explanations = {item["customer_id"]: item for item in parsed.get("explanations", [])}
        
        explained = []
        for item in top_entities:
            cust_id = str(item.get("customer_id"))
            if cust_id in llm_explanations:
                item_copy = dict(item)
                item_copy["explanation"] = llm_explanations[cust_id].get("explanation")
                item_copy["escalation_action"] = llm_explanations[cust_id].get("escalation_action")
                explained.append(item_copy)
            else:
                # Fallback item explanation if missing from LLM response
                tmpl = _generate_template_explanations([item])[0]
                explained.append(tmpl)
                
        context["explanations"] = explained
        return context

    except Exception as e:
        print(f"[!] Groq explanation API call failed: {e}. Using template explanation generator.")
        context["explanations"] = _generate_template_explanations(top_entities)
        return context
