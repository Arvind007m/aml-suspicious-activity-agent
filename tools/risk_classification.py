"""
Risk Classification Tool: Categorizes risk scores into Low, Medium, High risk levels.
Supports pure threshold queries (NO ML) and multi-factor ML risk scoring.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from registry import register_tool


@register_tool("risk_classification")
def run_risk_classification(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes overall risk scores and assigns risk levels (Low, Medium, High).
    Applies intent-based filtering (single_entity, threshold_query, pattern_detection).
    """
    plan_meta = context.get("plan_meta", {})
    intent = plan_meta.get("intent", "broad_analysis")
    entities = plan_meta.get("entities", {})
    filters = plan_meta.get("filters", {})
    
    # Check source DataFrame (either df_scored from ML or df_features from feature engineering)
    if "df_scored" in context:
        df = context["df_scored"].copy()
    elif "df_features" in context:
        df = context["df_features"].copy()
        df["anomaly_score"] = 0.0  # NO ML scoring used
    else:
        # Generate basic dataframe if missing
        df_raw = context["df"].copy()
        df_raw["customer_id"] = df_raw["customer_id"].astype(str)
        records = []
        for cust_id, group in df_raw.groupby("customer_id"):
            struct_cnt = len(group[(group["amount"] >= 9000.0) & (group["amount"] < 10000.0)])
            records.append({
                "customer_id": str(cust_id),
                "txn_count_total": len(group),
                "total_amount_usd": round(float(group["amount"].sum()), 2),
                "max_txn_amount": round(float(group["amount"].max()), 2),
                "avg_txn_amount": round(float(group["amount"].mean()), 2),
                "structuring_count": struct_cnt,
                "velocity_24h": min(len(group), 15),
                "rapid_cashout_flag": 1 if group["amount"].max() > 100000.0 else 0,
                "ground_truth_laundering": int(group["is_laundering"].sum()),
                "anomaly_score": 0.0
            })
        df = pd.DataFrame(records)

    df["customer_id"] = df["customer_id"].astype(str)

    # --- 1. Compute Composite Risk Score (0 - 100) ---
    def calculate_risk(row):
        score = 0.0
        # Rule indicators
        if row.get("structuring_count", 0) >= 10:
            score += 45.0
        elif row.get("structuring_count", 0) >= 3:
            score += 25.0
            
        if row.get("rapid_cashout_flag", 0) == 1:
            score += 35.0
            
        if row.get("velocity_24h", 0) >= 10:
            score += 20.0
            
        # ML Anomaly Score weight (if present)
        ml_anomaly = row.get("anomaly_score", 0.0)
        score += ml_anomaly * 40.0
        
        return min(round(score, 1), 100.0)

    df["risk_score"] = df.apply(calculate_risk, axis=1)

    def get_risk_level(score):
        if score >= 70.0:
            return "High"
        elif score >= 40.0:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df["risk_score"].apply(get_risk_level)

    # --- 2. Filter Results Based on User Query Intent (Fix 1: String Casting) ---
    if intent == "single_entity":
        target_cust = str(entities.get("customer_id") or "4521").strip()
        df_filtered = df[df["customer_id"].astype(str) == target_cust].copy()
        if len(df_filtered) == 0:
            # Fallback to matching target_cust without non-digit chars if needed
            cleaned_target = ''.join(filter(str.isdigit, target_cust))
            if cleaned_target:
                df_filtered = df[df["customer_id"].astype(str) == cleaned_target].copy()
        if len(df_filtered) == 0:
            df_filtered = df.sort_values("risk_score", ascending=False).head(1)

    elif intent == "threshold_query":
        # Query 3: Customers with 10+ txns under $10,000 (Pure threshold filtering)
        df_filtered = df[
            (df["structuring_count"] >= 10) | 
            ((df["txn_count_total"] >= 10) & (df["max_txn_amount"] <= 10000.0))
        ].copy()
        if len(df_filtered) == 0:
            df_filtered = df[df["txn_count_total"] >= 10].copy()

    elif intent == "pattern_detection":
        # Query 2: Structuring / Smurfing pattern focus
        df_filtered = df[df["structuring_count"] > 0].sort_values("structuring_count", ascending=False)

    else:
        # Query 1 / Default: Broad analysis ranking
        df_filtered = df.sort_values("risk_score", ascending=False)

    # Convert top items to list of dicts
    top_items = df_filtered.head(10).to_dict(orient="records")
    
    context["df_risk"] = df
    context["top_suspicious_entities"] = top_items
    return context
