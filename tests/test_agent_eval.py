"""
Trace-Level Evaluation Suite (tests/test_agent_eval.py)
Asserts deterministic planner tool-selection decisions, detection accuracy, and scoring determinism.
Offline-safe evaluation runner using rule-based planner.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from planner import _rule_based_fallback_planner, create_plan
from orchestrator import run_agent_query


def test_tool_selection_decisions():
    """Asserts that planner produces canonical plan and skipped arrays for core queries."""
    
    # 1. Broad Analysis Query
    p1 = _rule_based_fallback_planner("Analyse this dataset for suspicious activity")
    assert p1["intent"] == "broad_analysis"
    assert p1["plan"] == ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"]
    assert p1["skipped"] == []

    # 2. Structuring Pattern Query
    p2 = _rule_based_fallback_planner("Find structuring patterns in the last 30 days")
    assert p2["intent"] == "pattern_detection"
    assert p2["skipped"] == ["eda"]

    # 3. Threshold Aggregation Query (NO ML!)
    p3 = _rule_based_fallback_planner("Which customers made 10+ transactions under $10,000?")
    assert p3["intent"] == "threshold_query"
    assert p3["skipped"] == ["eda", "anomaly_detection"]

    # 4. Single Entity Lookup Query
    p4 = _rule_based_fallback_planner("Is customer 4521 suspicious?")
    assert p4["intent"] == "single_entity"
    assert p4["skipped"] == ["eda", "feature_engineering"]


def test_detection_accuracy():
    """Asserts that known laundering entities (3310, 1089, 4521) are correctly caught."""
    
    # Broad Analysis must catch 3310, 1089, 4521
    ctx_broad = run_agent_query("Analyse this dataset for suspicious activity")
    top_broad = ctx_broad.get("explanations", [])
    high_broad_custs = set(e["customer_id"] for e in top_broad if e.get("risk_level") == "High")
    assert "3310" in high_broad_custs
    assert "1089" in high_broad_custs
    assert "4521" in high_broad_custs

    # Single Entity 4521 lookup must return High risk
    ctx_4521 = run_agent_query("Is customer 4521 suspicious?")
    top_4521 = ctx_4521.get("explanations", [])
    assert len(top_4521) > 0
    assert top_4521[0]["customer_id"] == "4521"
    assert top_4521[0]["risk_level"] == "High"

    # Rapid Cash Out query must return Customer 3310
    ctx_cashout = run_agent_query("Show me customers who received large deposits then emptied their account within an hour")
    top_cashout = ctx_cashout.get("explanations", [])
    assert len(top_cashout) > 0
    assert top_cashout[0]["customer_id"] == "3310"
    assert top_cashout[0]["risk_level"] == "High"


def test_scoring_determinism():
    """Asserts that running the same query 3 times yields 100% identical risk scores."""
    q = "Analyse this dataset for suspicious activity"
    scores_run1 = [e["risk_score"] for e in run_agent_query(q).get("explanations", [])]
    scores_run2 = [e["risk_score"] for e in run_agent_query(q).get("explanations", [])]
    scores_run3 = [e["risk_score"] for e in run_agent_query(q).get("explanations", [])]

    assert scores_run1 == scores_run2
    assert scores_run2 == scores_run3


def test_non_existent_customer_handling():
    """Asserts that querying a non-existent customer (e.g. 99999) halts tool execution without running tools."""
    ctx = run_agent_query("Is customer 99999 suspicious?")
    plan_meta = ctx.get("plan_meta", {})
    assert plan_meta.get("intent") == "customer_not_found"
    assert ctx.get("executed_tools") == []
    assert len(ctx.get("explanations", [])) == 0



def run_standalone_eval():
    """Standalone runner for plain python tests/test_agent_eval.py execution."""
    print("\n" + "="*70)
    print("        RUNNING TRACE-LEVEL AGENT EVALUATION SUITE            ")
    print("="*70 + "\n")

    tool_checks_passed = 0
    try:
        test_tool_selection_decisions()
        tool_checks_passed = 6  # 4 plans + 2 intent checks
        print("  [+] Tool Selection Evals: 6/6 checks passed")
    except Exception as e:
        print(f"  [!] Tool Selection Evals failed: {e}")

    detection_checks_passed = 0
    try:
        test_detection_accuracy()
        detection_checks_passed = 3
        print("  [+] Detection Accuracy Evals: 3/3 checks passed")
    except Exception as e:
        print(f"  [!] Detection Accuracy Evals failed: {e}")

    determinism_status = "FAIL"
    try:
        test_scoring_determinism()
        determinism_status = "OK"
        print("  [+] Scoring Determinism Check: determinism OK")
    except Exception as e:
        print(f"  [!] Scoring Determinism Check failed: {e}")

    print("\n" + "="*70)
    print(f"  SUMMARY: {tool_checks_passed}/6 tool-selection checks passed, {detection_checks_passed}/3 detection checks passed, determinism {determinism_status}")
    print("="*70 + "\n")

    if tool_checks_passed == 6 and detection_checks_passed == 3 and determinism_status == "OK":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(run_standalone_eval())
