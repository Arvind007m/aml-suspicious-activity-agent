"""
Central Configuration & Domain Thresholds Module (config.py)
Defines named, defensible AML domain constants, risk weights, and model parameters.
"""

# --- AML Compliance & Threshold Rules ---
STRUCTURING_LOWER_THRESHOLD = 9000.0   # Lower bound for near-threshold cash deposits
CTR_REPORTING_THRESHOLD = 10000.0      # Currency Transaction Report (CTR) regulatory threshold
LARGE_INBOUND_THRESHOLD = 100000.0     # Threshold for large inbound wire deposit in rapid cash-out
CASHOUT_MIN_AMOUNT_FILTER = 50000.0    # Min transaction filter for rapid cash-out queries
CASHOUT_DRAIN_PCT_THRESHOLD = 70.0      # Min percentage of inbound deposit drained in rapid cash-out
CASHOUT_WINDOW_MINUTES = 45             # Time window (minutes) for rapid split outflows
VELOCITY_24H_THRESHOLD = 10            # High transaction count velocity threshold in rolling 24h

# --- Machine Learning Model Parameters ---
ISO_CONTAMINATION = 0.03                # IsolationForest anomaly contamination factor
RANDOM_SEED = 42                        # Determinism random seed

# --- Risk Scoring Weights & Thresholds ---
RISK_HIGH_THRESHOLD = 70.0              # Risk score lower bound for High risk level
RISK_MEDIUM_THRESHOLD = 40.0            # Risk score lower bound for Medium risk level
CONFIDENCE_HIGH_SCORE = 75.0            # Score threshold for High confidence
CONFIDENCE_MEDIUM_SCORE = 40.0          # Score threshold for Medium confidence

# --- Guardrails ---
MAX_QUERY_LENGTH = 500                  # Maximum allowed search query length
MAX_TOOLS_EXECUTION_CAP = 5             # Compounding error guardrail tool cap
