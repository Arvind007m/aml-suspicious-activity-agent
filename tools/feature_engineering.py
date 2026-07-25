"""
Feature Engineering Tool: Aggregates velocity, structuring, rolling windows, and cashout indicators per customer.
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
    """
    df: pd.DataFrame = context["df"].copy()
    filters = context.get("plan_meta", {}).get("filters", {})
    
    # Apply date range filter if specified in plan
    df["dt"] = pd.to_datetime(df["timestamp"])
    max_dt = df["dt"].max()
    date_days = filters.get("date_range_days")
    if date_days:
        cutoff = max_dt - timedelta(days=int(date_days))
        df = df[df["dt"] >= cutoff].copy()

    # Per-customer feature calculation
    records = []
    for cust_id, group in df.groupby("customer_id"):
        group = group.sort_values("dt")
        
        txn_count = len(group)
        total_amt = group["amount"].sum()
        max_amt = group["amount"].max()
        avg_amt = group["amount"].mean()
        
        # Structuring check: transactions between $9,000 and $9,999
        structuring_txns = group[(group["amount"] >= 9000.0) & (group["amount"] < 10000.0)]
        structuring_count = len(structuring_txns)
        
        # Calculate max rolling 24h transaction velocity
        max_24h_velocity = 1
        if txn_count > 1:
            # Rolling count over 24h window
            window_counts = group.set_index("dt").rolling("24h")["transaction_id"].count()
            max_24h_velocity = int(window_counts.max())
            
        # Rapid cash-out indicator: Large inbound > $100k followed by multiple outbound wires within 1h
        rapid_cashout = 0
        inbound_large = group[group["amount"] > 100000.0]
        if len(inbound_large) > 0 and txn_count > 5:
            rapid_cashout = 1

        laundering_count = int(group["is_laundering"].sum())

        records.append({
            "customer_id": str(cust_id),
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
    context["df_features"] = df_features
    return context
