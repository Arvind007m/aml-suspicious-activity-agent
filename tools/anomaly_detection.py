"""
Anomaly Detection Tool: Machine Learning (Isolation Forest) Anomaly Scoring.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from registry import register_tool


@register_tool("anomaly_detection")
def run_anomaly_detection(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fits IsolationForest ML model on customer feature vector to compute anomaly scores.
    """
    # If df_features is missing (e.g. single entity query skipping feature_engineering), compute basic features first
    if "df_features" not in context:
        df: pd.DataFrame = context["df"]
        records = []
        for cust_id, group in df.groupby("customer_id"):
            structuring_count = len(group[(group["amount"] >= 9000.0) & (group["amount"] < 10000.0)])
            records.append({
                "customer_id": str(cust_id),
                "txn_count_total": len(group),
                "total_amount_usd": round(float(group["amount"].sum()), 2),
                "max_txn_amount": round(float(group["amount"].max()), 2),
                "avg_txn_amount": round(float(group["amount"].mean()), 2),
                "structuring_count": structuring_count,
                "velocity_24h": min(len(group), 15),
                "rapid_cashout_flag": 1 if group["amount"].max() > 100000.0 else 0,
                "ground_truth_laundering": int(group["is_laundering"].sum())
            })
        df_features = pd.DataFrame(records)
    else:
        df_features = context["df_features"].copy()

    feature_cols = [
        "structuring_count",
        "velocity_24h",
        "total_amount_usd",
        "max_txn_amount",
        "rapid_cashout_flag"
    ]
    
    X = df_features[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X_scaled)
    
    # Calculate decision function (lower values = more anomalous)
    raw_scores = model.decision_function(X_scaled)
    
    # Normalize score to [0.0, 1.0] where 1.0 is highest anomaly risk
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        normalized_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
    else:
        normalized_scores = np.zeros(len(raw_scores))
        
    df_features["anomaly_score"] = np.round(normalized_scores, 4)
    context["df_scored"] = df_features
    return context
