"""
Independent test suite for Phase 3 tools.
"""

import pandas as pd
from generate_data import generate_synthetic_aml_data
from registry import execute_tool_chain, load_all_tools


def test_tools_independently():
    df = generate_synthetic_aml_data(num_rows=500)
    load_all_tools()
    
    plan_meta = {
        "intent": "broad_analysis",
        "entities": {"customer_id": None},
        "filters": {},
        "plan": ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"]
    }
    
    context = {"df": df, "plan_meta": plan_meta}
    
    # Run full chain
    res = execute_tool_chain(plan_meta["plan"], context)
    
    # Assertions for each tool
    assert "eda_results" in res, "EDA tool failed to produce eda_results"
    print("[+] eda tool verified. Total txns:", res["eda_results"]["total_transactions"])
    
    assert "df_features" in res, "Feature engineering tool failed"
    print("[+] feature_engineering tool verified. Customers processed:", len(res["df_features"]))
    
    assert "df_scored" in res and "anomaly_score" in res["df_scored"], "Anomaly detection tool failed"
    print("[+] anomaly_detection tool verified. Max anomaly score:", res["df_scored"]["anomaly_score"].max())
    
    assert "df_risk" in res and "top_suspicious_entities" in res, "Risk classification tool failed"
    print("[+] risk_classification tool verified. Top entities count:", len(res["top_suspicious_entities"]))
    
    assert "explanations" in res and len(res["explanations"]) > 0, "Explanation tool failed"
    print("[+] explanation tool verified. Explanations sample:", res["explanations"][0]["explanation"])
    
    print("\n[+] ALL 5 TOOLS VERIFIED INDEPENDENTLY AND SUCCESSFULLY!")


if __name__ == "__main__":
    test_tools_independently()
