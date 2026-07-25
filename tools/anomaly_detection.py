"""
Anomaly Detection Tool: Machine Learning (Isolation Forest) Anomaly Scoring.
Strictly prevents label leakage by excluding ground-truth 'is_laundering' and string ID columns.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from registry import register_tool
from tools.feature_engineering import compute_customer_features
from config import ISO_CONTAMINATION, RANDOM_SEED


@register_tool("anomaly_detection")
def run_anomaly_detection(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fits IsolationForest ML model on customer feature vector to compute anomaly scores.
    Strictly trains ONLY on numeric engineered behavioral features to prevent label leakage.
    """
    # Use unified compute_customer_features if df_features was skipped (Fix C3)
    if "df_features" not in context or context["df_features"].empty:
        plan_meta = context.get("plan_meta", {})
        df_features = compute_customer_features(context["df"], plan_meta)
    else:
        df_features = context["df_features"].copy()

    df_features["customer_id"] = df_features["customer_id"].astype(str)

    # ML Feature Matrix (Fix 3): ONLY numeric behavioral features feed the IsolationForest model.
    # EXPLICITLY EXCLUDED: is_laundering (ground truth label), ground_truth_laundering, customer_id, transaction_id, accounts, currency, timestamps.
    feature_cols = [
        "structuring_count",
        "velocity_24h",
        "total_amount_usd",
        "max_txn_amount",
        "rapid_cashout_flag"
    ]
    
    print(f"  [ML Model Features Trained On]: {feature_cols} (is_laundering EXCLUDED)")
    
    X = df_features[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest model using centralized parameters (Fix M1)
    model = IsolationForest(contamination=ISO_CONTAMINATION, random_state=RANDOM_SEED)
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
