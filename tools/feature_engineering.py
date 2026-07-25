from typing import Dict, Any
from datetime import timedelta
import pandas as pd
import numpy as np
from registry import register_tool
from config import (
    STRUCTURING_LOWER_THRESHOLD,
    CTR_REPORTING_THRESHOLD,
    LARGE_INBOUND_THRESHOLD,
    CASHOUT_DRAIN_PCT_THRESHOLD
)



def compute_customer_features(df: pd.DataFrame, plan_meta: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Unified Single Source of Truth for computing per-customer behavioral features (C3).
    Used by feature_engineering tool and on-demand by anomaly_detection and risk_classification.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "customer_id", "txn_count_total", "total_amount_usd", "max_txn_amount",
            "min_txn_amount", "avg_txn_amount", "structuring_count", "structuring_min_amt",
            "structuring_max_amt", "velocity_24h", "rapid_cashout_flag", "inbound_deposit_usd",
            "outbound_drained_usd", "cashout_drained_pct", "outbound_txn_count",
            "cashout_window_minutes", "ground_truth_laundering"
        ])

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"])

    records = []
    for cust_id, group in df.groupby("customer_id"):
        group = group.sort_values("dt")
        cust_id_str = str(cust_id)

        txn_count = len(group)
        total_amt = group["amount"].sum() if txn_count > 0 else 0.0
        max_amt = group["amount"].max() if txn_count > 0 else 0.0
        min_amt = group["amount"].min() if txn_count > 0 else 0.0
        avg_amt = group["amount"].mean() if txn_count > 0 else 0.0

        # Pattern 1: Structuring check ($9,000 - $10,000)
        structuring_txns = group[
            (group["amount"] >= STRUCTURING_LOWER_THRESHOLD) &
            (group["amount"] < CTR_REPORTING_THRESHOLD)
        ]
        structuring_count = len(structuring_txns)
        struct_min = float(structuring_txns["amount"].min()) if structuring_count > 0 else 0.0
        struct_max = float(structuring_txns["amount"].max()) if structuring_count > 0 else 0.0

        # Pattern 2: Rapid cash-out check
        rapid_cashout = 0
        inbound_deposit_val = 0.0
        outbound_drained_val = 0.0
        drained_pct = 0.0
        outbound_cnt = 0
        window_mins = 0

        large_inbound = group[(group["payment_format"] == "Wire") & (group["amount"] >= LARGE_INBOUND_THRESHOLD)]
        if len(large_inbound) > 0:
            inbound_row = large_inbound.iloc[0]
            inbound_deposit_val = float(inbound_row["amount"])
            inbound_time = inbound_row["dt"]

            # Outbound transfers within 2 hours
            outbound_txns = group[(group["dt"] > inbound_time) & (group["dt"] <= inbound_time + timedelta(hours=2))]
            outbound_cnt = len(outbound_txns)
            if outbound_cnt >= 3:
                outbound_drained_val = float(outbound_txns["amount"].sum())
                drained_pct = (outbound_drained_val / inbound_deposit_val) * 100.0 if inbound_deposit_val > 0 else 0.0
                last_outbound_time = outbound_txns["dt"].max()
                window_mins = int((last_outbound_time - inbound_time).total_seconds() / 60.0)
                if drained_pct >= CASHOUT_DRAIN_PCT_THRESHOLD:
                    rapid_cashout = 1

        # Max rolling 24h transaction velocity
        max_24h_velocity = 1
        if txn_count > 1:
            window_counts = group.set_index("dt").rolling("24h")["transaction_id"].count()
            max_24h_velocity = int(window_counts.max())

        laundering_count = int(group["is_laundering"].sum()) if "is_laundering" in group.columns else 0

        records.append({
            "customer_id": cust_id_str,
            "txn_count_total": txn_count,
            "total_amount_usd": round(float(total_amt), 2),
            "max_txn_amount": round(float(max_amt), 2),
            "min_txn_amount": round(float(min_amt), 2),
            "avg_txn_amount": round(float(avg_amt), 2),
            "structuring_count": structuring_count,
            "structuring_min_amt": round(struct_min, 2),
            "structuring_max_amt": round(struct_max, 2),
            "velocity_24h": max_24h_velocity,
            "rapid_cashout_flag": rapid_cashout,
            "inbound_deposit_usd": round(inbound_deposit_val, 2),
            "outbound_drained_usd": round(outbound_drained_val, 2),
            "cashout_drained_pct": round(drained_pct, 1),
            "outbound_txn_count": outbound_cnt,
            "cashout_window_minutes": window_mins if window_mins > 0 else 45,
            "ground_truth_laundering": laundering_count
        })

    df_features = pd.DataFrame(records)
    if not df_features.empty:
        df_features["customer_id"] = df_features["customer_id"].astype(str)
    return df_features


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

    df["dt"] = pd.to_datetime(df["timestamp"])
    orig_rows = len(df)

    # 1. Apply date range filter if specified in query plan (Fix H4)
    date_days = filters.get("date_range_days")
    if date_days:
        max_dt = df["dt"].max()
        cutoff = max_dt - timedelta(days=int(date_days))
        df = df[df["dt"] >= cutoff].copy()
        if len(df) == 0:
            print(f"  [!] No transactions found in the last {date_days} days window.")
            context["df_features"] = pd.DataFrame()
            return context

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

    if len(df) == 0:
        print(f"  [!] No transactions match the filter criteria.")
        context["df_features"] = pd.DataFrame()
        return context

    context["df_features"] = compute_customer_features(df, plan_meta)
    return context

