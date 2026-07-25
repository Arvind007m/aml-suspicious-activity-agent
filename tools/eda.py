"""
EDA Tool: Exploratory Data Analysis and Dataset Profiling.
"""

from typing import Dict, Any
import pandas as pd
from registry import register_tool


@register_tool("eda")
def run_eda(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profiles dataset stats, missing values, transaction format breakdown, and time span.
    """
    df: pd.DataFrame = context["df"]
    
    total_txns = len(df)
    total_volume = df["amount"].sum()
    unique_customers = df["customer_id"].nunique()
    unique_senders = df["sender_account"].nunique()
    unique_receivers = df["receiver_account"].nunique()
    missing_count = int(df.isnull().sum().sum())
    
    df["dt"] = pd.to_datetime(df["timestamp"])
    min_date = df["dt"].min().strftime("%Y-%m-%d")
    max_date = df["dt"].max().strftime("%Y-%m-%d")
    
    payment_counts = df["payment_format"].value_counts().to_dict()
    laundering_ground_truth_count = int(df["is_laundering"].sum())
    
    eda_summary = {
        "total_transactions": total_txns,
        "total_volume_usd": round(float(total_volume), 2),
        "unique_customers": unique_customers,
        "unique_senders": unique_senders,
        "unique_receivers": unique_receivers,
        "missing_values": missing_count,
        "date_range": f"{min_date} to {max_date}",
        "payment_formats": payment_counts,
        "ground_truth_laundering_txns": laundering_ground_truth_count
    }
    
    context["eda_results"] = eda_summary
    return context
