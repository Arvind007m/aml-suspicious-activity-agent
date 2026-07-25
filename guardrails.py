"""
Input Validation & Security Guardrails Module (guardrails.py)
Validates query input before planning to prevent empty inputs, query length abuse, and malicious patterns.
"""

import re
from typing import Tuple, Dict, Any

# Disallowed malicious & command injection keywords
DISALLOWED_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\brm\s+-rf\b",
    r"\bsystem\s*\(",
    r"<\s*script",
    r"\bunion\s+select\b",
    r"\bexec\s*\(",
    r"\beval\s*\("
]


def validate_query(query: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates user query input before passing to planner or tool pipeline.
    
    Returns:
        (is_valid: bool, error_message: str, fallback_plan: dict or None)
    """
    raw_str = str(query).strip()
    q_str = raw_str.strip('"').strip("'").strip()

    # 1. Empty or whitespace/quotation-only query guard
    if not q_str:
        plan = {
            "planner_type": "Input Guardrail",
            "intent": "needs_clarification",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": [],
            "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "reason": "Empty query received. Please enter a search query.",
            "clarifying_question": "Please enter a valid search query (e.g. 'Analyse this dataset' or 'Is customer 4521 suspicious?').",
            "reasoning_trace": [
                "Guardrail check -> EMPTY / QUOTE-ONLY QUERY REJECTED",
                "Decision: HALT execution & prompt user for valid query"
            ]
        }
        return False, "Query cannot be empty. Please enter a search query.", plan


    # 2. Maximum query length guard (prevent long text abuse)
    if len(q_str) > 500:
        plan = {
            "planner_type": "Input Guardrail",
            "intent": "needs_clarification",
            "entities": {"customer_id": None},
            "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
            "aml_pattern": None,
            "plan": [],
            "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
            "reason": "Query exceeds maximum length limit of 500 characters.",
            "clarifying_question": "Your query is too long (over 500 characters). Please shorten your query.",
            "reasoning_trace": [
                f"Guardrail check -> QUERY LENGTH ({len(q_str)} chars) EXCEEDS 500 CHAR LIMIT",
                "Decision: HALT execution & prompt user to shorten query"
            ]
        }
        return False, "Query exceeds maximum length of 500 characters. Please enter a shorter query.", plan

    # 3. Malicious pattern & injection guard
    q_lower = q_str.lower()
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, q_lower):
            plan = {
                "planner_type": "Input Guardrail",
                "intent": "needs_clarification",
                "entities": {"customer_id": None},
                "filters": {"date_range_days": None, "min_amount": None, "max_amount": None, "min_txn_count": None},
                "aml_pattern": None,
                "plan": [],
                "skipped": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"],
                "reason": "Query refused: disallowed or malicious pattern detected.",
                "clarifying_question": "Security Guardrail Alert: Query contains a disallowed or potentially malicious pattern.",
                "reasoning_trace": [
                    f"Guardrail check -> MALICIOUS PATTERN DETECTED ('{pattern}')",
                    "Decision: REFUSE EXECUTION for security"
                ]
            }
            return False, "Query refused: disallowed or malicious pattern detected.", plan

    return True, "", None
