"""
Feature Engineering Tool: Aggregates velocity, structuring, rolling windows, and cashout indicators per customer.
Applies date range, min/max amount, and count filtering extracted from user queries.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import timedelta
from registry import register_tool


@register_tool("feature_engineering")
def run_feature_engineering(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes per-customer behavioral features for AML detection.
    Applies query filters (date_range_days, min_amount, max_amount) before feature extraction.
    """
    df: pd.DataFrame = context["df"].copy()
    df["customer_id"] = df["customer_id"].astype(str)
    filters = context.get("plan_meta", {}).get("filters", {})
    
    orig_rows = len(df)
    
    # 1. Apply date range filter if specified in query plan (Fix 2)
    df["dt"] = pd.to_datetime(df["timestamp"])
    max_dt = df["dt"].max()
    date_days = filters.get("date_range_days")
    if date_days:
        cutoff = max_dt - timedelta(days=int(date_days))
        df = df[df["dt"] >= cutoff].copy()

    # 2. Apply amount filters if specified in query plan
    min_amt = filters.get("min_amount")
    if min_amt is not None:
        df = df[df["amount"] >= float(min_amt)]
        
    max_amt = filters.get("max_amount")
    if max_amt is not None:
        df = df[df["amount"] <= float(max_amt)]

    filtered_rows = len(df)
    if date_days or min_amt or max_amt:
        print(f"  [Feature Engineering Filter Applied]: {filtered_rows} / {orig_rows} rows remaining after filtering.")

    # Per-customer feature calculation
    records = []
    for cust_id, group in df.groupby("customer_id"):
        group = group.sort_values("dt")
        cust_id_str = str(cust_id)
        
        txn_count = len(group)
        total_amt = group["amount"].sum() if txn_count > 0 else 0.0
        max_amt = group["amount"].max() if txn_count > 0 else 0.0
        avg_amt = group["amount"].mean() if txn_count > 0 else 0.0
        
        # Structuring check: transactions between $9,000 and $9,999
        structuring_txns = group[(group["amount"] >= 9000.0) & (group["amount"] < 10000.0)]
        structuring_count = len(structuring_txns)
        
        # Calculate max rolling 24h transaction velocity
        max_24h_velocity = 1
        if txn_count > 1:
            window_counts = group.set_index("dt").rolling("24h")["transaction_id"].count()
            max_24h_velocity = int(window_counts.max())
            
        # Rapid cash-out indicator: Large inbound > $100k followed by multiple outbound wires within short window
        rapid_cashout = 0
        inbound_large = group[group["amount"] > 100000.0]
        if len(inbound_large) > 0 and txn_count > 5:
            rapid_cashout = 1

        laundering_count = int(group["is_laundering"].sum())

        records.append({
            "customer_id": cust_id_str,
            "txn_count_total": txn_count,
            "total_amount_usd": round(float(total_amt), 2),
            "max_txn_amount": round(float(max_amt), 2),
            "avg_txn_amount": round(float(avg_amt), 2),
            "structuring_count": structuring_count,
            "velocity_24h": max_24h_velocity,
            "rapid_cashout_flag": rapid_cashout,
            "ground_truth_laundering": laundering_count
        })

    df_features = pd.DataFrame(records)
    if len(df_features) > 0:
        df_features["customer_id"] = df_features["customer_id"].astype(str)

    context["df_features"] = df_features
    return context
