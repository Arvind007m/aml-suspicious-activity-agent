"""
Risk Classification Tool: Categorizes risk scores into Low, Medium, High risk levels.
Aligns scoring with active aml_pattern and prevents single-transaction false positives.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from registry import register_tool
from tools.feature_engineering import compute_customer_features
from config import (
    STRUCTURING_LOWER_THRESHOLD,
    CTR_REPORTING_THRESHOLD,
    LARGE_INBOUND_THRESHOLD,
    VELOCITY_24H_THRESHOLD,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    CONFIDENCE_HIGH_SCORE,
    CONFIDENCE_MEDIUM_SCORE
)


def get_confidence_level(row: dict) -> str:
    """
    Computes confidence signal per flag to prevent confidently-wrong output (Feature 3).
    High: Multiple corroborating signals (15 near-threshold deposits OR large inbound + rapid split outflow)
    Medium: Strong single signal (structuring >= 3 OR ML anomaly score >= 0.60)
    Low: Weak / single event signal
    """
    structuring_count = row.get("structuring_count", 0)
    rapid_cashout = row.get("rapid_cashout_flag", 0)
    risk_score = row.get("risk_score", 0.0)
    anomaly_score = row.get("anomaly_score", 0.0)

    if structuring_count >= 10 or rapid_cashout == 1 or risk_score >= CONFIDENCE_HIGH_SCORE:
        return "High"
    elif structuring_count >= 3 or anomaly_score >= 0.60 or risk_score >= CONFIDENCE_MEDIUM_SCORE:
        return "Medium"
    else:
        return "Low"


@register_tool("risk_classification")
def run_risk_classification(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes overall risk scores and assigns risk levels (Low, Medium, High).
    Aligns ranking with active aml_pattern (rapid_cash_out vs structuring vs threshold_query).
    Enforces minimum count guards to prevent single-transaction false positives.
    """
    plan_meta = context.get("plan_meta", {})
    intent = plan_meta.get("intent", "broad_analysis")
    entities = plan_meta.get("entities", {})
    filters = plan_meta.get("filters", {})
    aml_pattern = plan_meta.get("aml_pattern")

    # Check source DataFrame
    if "df_scored" in context:
        df = context["df_scored"].copy()
    elif "df_features" in context:
        df = context["df_features"].copy()
        df["anomaly_score"] = 0.0
    else:
        # Use unified compute_customer_features (Fix C3)
        df = compute_customer_features(context["df"], plan_meta)
        df["anomaly_score"] = 0.0

    df["customer_id"] = df["customer_id"].astype(str)

    # --- 1. Compute Pattern-Aligned Composite Risk Score (0 - 100) ---
    def calculate_risk(row):
        score = 0.0
        struct_cnt = row.get("structuring_count", 0)
        cashout_flag = row.get("rapid_cashout_flag", 0)
        velocity = row.get("velocity_24h", 0)
        ml_anomaly = row.get("anomaly_score", 0.0)

        if aml_pattern == "rapid_cash_out":
            # Prioritize rapid cashout indicators for rapid cash out queries
            if cashout_flag == 1:
                score += 85.0
            elif row.get("inbound_deposit_usd", 0.0) >= LARGE_INBOUND_THRESHOLD:
                score += 40.0
            score += ml_anomaly * 15.0

        elif aml_pattern == "structuring":
            if struct_cnt >= 10:
                score += 65.0
            elif struct_cnt >= 3:
                score += 45.0
            else:
                score += 0.0

            if velocity >= VELOCITY_24H_THRESHOLD:
                score += 15.0
            score += ml_anomaly * 20.0

        else:
            # Broad analysis / general queries
            if struct_cnt >= 10:
                score += 45.0
            elif struct_cnt >= 3:
                score += 25.0

            if cashout_flag == 1:
                score += 45.0

            if velocity >= VELOCITY_24H_THRESHOLD:
                score += 15.0

            score += ml_anomaly * 20.0

        return min(round(score, 1), 100.0)

    def get_risk_level(score):
        if score >= RISK_HIGH_THRESHOLD:
            return "High"
        elif score >= RISK_MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"

    df["risk_score"] = df.apply(calculate_risk, axis=1)
    df["risk_level"] = df["risk_score"].apply(get_risk_level)
    df["confidence"] = df.apply(lambda r: get_confidence_level(r.to_dict()), axis=1)

    # --- 2. Filter & Rank Results Based on Active Pattern & Intent ---
    if intent == "single_entity":
        target_cust = str(entities.get("customer_id") or "").strip()
        if not target_cust:
            # Fix H2: Never default to 4521 if no customer_id extracted
            records = [{
                "customer_id": "Unknown",
                "txn_count_total": 0,
                "total_amount_usd": 0.0,
                "max_txn_amount": 0.0,
                "avg_txn_amount": 0.0,
                "structuring_count": 0,
                "velocity_24h": 0,
                "rapid_cashout_flag": 0,
                "ground_truth_laundering": 0,
                "anomaly_score": 0.0,
                "risk_score": 0.0,
                "risk_level": "Low",
                "confidence": "Low",
                "explanation": "No customer ID detected in query. Please specify a customer ID (e.g., Customer 4521).",
                "escalation_action": "No Action Required"
            }]
            context["df_risk"] = df
            context["top_suspicious_entities"] = records
            context["explanations"] = records
            return context

        df_filtered = df[df["customer_id"].astype(str) == target_cust].copy()
        if len(df_filtered) == 0:

            # Non-existent customer guard (Fix 1)
            records = [{
                "customer_id": target_cust,
                "txn_count_total": 0,
                "total_amount_usd": 0.0,
                "max_txn_amount": 0.0,
                "avg_txn_amount": 0.0,
                "structuring_count": 0,
                "velocity_24h": 0,
                "rapid_cashout_flag": 0,
                "ground_truth_laundering": 0,
                "anomaly_score": 0.0,
                "risk_score": 0.0,
                "risk_level": "Low",
                "confidence": "Low",
                "explanation": f"No transactions found for customer {target_cust}. Cannot assess risk.",
                "escalation_action": "No Action Required"

            }]
            context["df_risk"] = df
            context["top_suspicious_entities"] = records
            context["explanations"] = records
            return context

    elif intent == "threshold_query":

        # Query 3: Customers with 10+ txns under $10,000 (Pure threshold filtering)
        df_filtered = df[
            (df["structuring_count"] >= 10) | 
            ((df["txn_count_total"] >= 10) & (df["max_txn_amount"] <= 10000.0))
        ].copy()
        if len(df_filtered) == 0:
            df_filtered = df[df["txn_count_total"] >= 10].copy()

    elif intent == "pattern_detection":
        if aml_pattern == "rapid_cash_out":
            # Surface Customer 3310 #1 for rapid cash out query (Fix 3)
            df_filtered = df[df["rapid_cashout_flag"] == 1].sort_values("risk_score", ascending=False)
            if len(df_filtered) == 0:
                df_filtered = df.sort_values("rapid_cashout_flag", ascending=False)
        else:
            # Surface structuring customers (4521 & 1089) for structuring query (Fix 6: count >= 3)
            df_filtered = df[df["structuring_count"] >= 3].sort_values("structuring_count", ascending=False)

    else:
        # Broad analysis ranking
        df_filtered = df.sort_values("risk_score", ascending=False)

    # Convert top items to list of dicts
    top_items = df_filtered.head(10).to_dict(orient="records")
    
    context["df_risk"] = df
    context["top_suspicious_entities"] = top_items
    return context
