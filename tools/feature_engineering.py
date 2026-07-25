"""
Feature Engineering Tool: Aggregates velocity, structuring, rolling windows, and cashout indicators per customer.
Branches feature calculation based on active aml_pattern (structuring, rapid_cash_out, amount_spike).
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
    Branches feature extraction based on plan_meta["aml_pattern"].
    """
    df: pd.DataFrame = context["df"].copy()
    df["customer_id"] = df["customer_id"].astype(str)
    plan_meta = context.get("plan_meta", {})
    filters = plan_meta.get("filters", {})
    aml_pattern = plan_meta.get("aml_pattern")
    
    orig_rows = len(df)
    
    # 1. Apply date range filter if specified in query plan
    df["dt"] = pd.to_datetime(df["timestamp"])
    max_dt = df["dt"].max()
    date_days = filters.get("date_range_days")
    if date_days:
        cutoff = max_dt - timedelta(days=int(date_days))
        df = df[df["dt"] >= cutoff].copy()

    # 2. Apply amount filters if specified in query plan
    min_amt = filters.get("min_amount")
    if min_amt is not None and aml_pattern != "rapid_cash_out":
        df = df[df["amount"] >= float(min_amt)]
        
    max_amt = filters.get("max_amount")
    if max_amt is not None:
        df = df[df["amount"] <= float(max_amt)]

    filtered_rows = len(df)
    if date_days or (min_amt and aml_pattern != "rapid_cash_out") or max_amt:
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
        
        # Pattern 1: Structuring check (transactions between $9,000 and $9,999)
        structuring_txns = group[(group["amount"] >= 9000.0) & (group["amount"] < 10000.0)]
        structuring_count = len(structuring_txns)
        
        # Pattern 2: Rapid cash-out check (inbound large deposit followed by rapid outbound split wires draining >70%)
        rapid_cashout = 0
        inbound_deposit_val = 0.0
        outbound_drained_val = 0.0
        drained_pct = 0.0
        outbound_cnt = 0
        
        # Check if customer has large inbound transfer
        large_inbound = group[(group["payment_format"] == "Wire") & (group["amount"] >= 100000.0)]
        if len(large_inbound) > 0:
            inbound_row = large_inbound.iloc[0]
            inbound_deposit_val = float(inbound_row["amount"])
            inbound_time = inbound_row["dt"]
            
            # Check outbound transfers within 2 hours after inbound deposit
            outbound_txns = group[(group["dt"] > inbound_time) & (group["dt"] <= inbound_time + timedelta(hours=2))]
            outbound_cnt = len(outbound_txns)
            if outbound_cnt >= 3:
                outbound_drained_val = float(outbound_txns["amount"].sum())
                drained_pct = (outbound_drained_val / inbound_deposit_val) * 100.0
                if drained_pct >= 70.0:
                    rapid_cashout = 1

        # Calculate max rolling 24h transaction velocity
        max_24h_velocity = 1
        if txn_count > 1:
            window_counts = group.set_index("dt").rolling("24h")["transaction_id"].count()
            max_24h_velocity = int(window_counts.max())
            
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
            "inbound_deposit_usd": round(inbound_deposit_val, 2),
            "outbound_drained_usd": round(outbound_drained_val, 2),
            "cashout_drained_pct": round(drained_pct, 1),
            "outbound_txn_count": outbound_cnt,
            "ground_truth_laundering": laundering_count
        })

    df_features = pd.DataFrame(records)
    if len(df_features) > 0:
        df_features["customer_id"] = df_features["customer_id"].astype(str)

    context["df_features"] = df_features
    return context
