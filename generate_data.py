"""
Synthetic IBM-AML-style transaction data generator.
Generates ~5,000 realistic banking transactions with baked-in laundering patterns:
1. Structuring / Smurfing (multiple transactions under $10,000 threshold)
2. Rapid Cash-Out / High Velocity (large deposit followed by immediate split wire-outs)
3. Amount Anomalies (sudden extreme transfer spikes)
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_synthetic_aml_data(num_rows: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic transaction dataset for AML detection.
    
    Columns:
    - timestamp (ISO 8601 string)
    - transaction_id (str)
    - customer_id (str) -> Primary entity key
    - sender_account (str)
    - receiver_account (str)
    - amount (float)
    - currency (str)
    - payment_format (str: Wire, ACH, Cash Deposit, Credit Card)
    - is_laundering (int: 0 or 1)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    start_date = datetime.now() - timedelta(days=90)
    
    # Generate base customer IDs and account pools
    customers = [f"CUST_{i:04d}" for i in range(1000, 1500)]
    # Explicitly include demo target customer ID 4521
    if "CUST_4521" not in customers:
        customers.append("CUST_4521")
        
    accounts = {cust: f"ACC_{cust[5:]}_01" for cust in customers}
    
    records = []
    current_time = start_date
    
    # --- 1. Normal Transactions (~4,700 rows) ---
    normal_rows = num_rows - 300
    for i in range(normal_rows):
        current_time += timedelta(seconds=random.randint(10, 1800))
        cust = random.choice(customers)
        sender_acc = accounts[cust]
        
        # Select external or internal receiver account
        receiver_cust = random.choice(customers)
        receiver_acc = accounts[receiver_cust] if receiver_cust != cust else f"EXT_ACC_{random.randint(5000, 9999)}"
        
        # Exponential distribution for normal amount (mostly small/medium $10-$2,500)
        amount = round(float(np.random.exponential(scale=350.0) + 15.0), 2)
        payment_fmt = random.choice(["ACH", "Credit Card", "Wire", "Cash Deposit"])
        currency = random.choice(["USD", "USD", "USD", "EUR", "GBP"])
        
        records.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": f"TXN_{100000 + i}",
            "customer_id": cust.replace("CUST_", ""),  # Clean ID format e.g. "4521", "1001"
            "sender_account": sender_acc,
            "receiver_account": receiver_acc,
            "amount": amount,
            "currency": currency,
            "payment_format": payment_fmt,
            "is_laundering": 0
        })

    # --- 2. Pattern 1: Structuring / Smurfing (Customer 4521 & Customer 1089) ---
    # Customer 4521: 15 cash deposits just under $10,000 within 24 hours
    structuring_custs = ["4521", "1089"]
    for cust_id in structuring_custs:
        base_time = start_date + timedelta(days=random.randint(40, 60))
        sender_acc = f"ACC_{cust_id}_01"
        for j in range(15):
            txn_time = base_time + timedelta(minutes=j * 35 + random.randint(1, 10))
            amount = round(random.uniform(9100.0, 9950.0), 2)  # Under $10k threshold
            records.append({
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_id": f"TXN_STRUCT_{cust_id}_{j}",
                "customer_id": cust_id,
                "sender_account": sender_acc,
                "receiver_account": f"SHELL_ACC_{random.randint(900, 999)}",
                "amount": amount,
                "currency": "USD",
                "payment_format": "Cash Deposit",
                "is_laundering": 1
            })

    # --- 3. Pattern 2: Rapid Cash-Out / High Velocity (Customer 3310) ---
    cust_id = "3310"
    base_time = start_date + timedelta(days=70)
    sender_acc = f"ACC_{cust_id}_01"
    # Large inbound
    records.append({
        "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id": f"TXN_INBOUND_{cust_id}",
        "customer_id": cust_id,
        "sender_account": "OFFSHORE_FUND_99",
        "receiver_account": sender_acc,
        "amount": 180000.0,
        "currency": "USD",
        "payment_format": "Wire",
        "is_laundering": 1
    })
    # Rapid outbound wires within 45 mins
    for j in range(12):
        txn_time = base_time + timedelta(minutes=j * 3 + 2)
        records.append({
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": f"TXN_OUTBOUND_{cust_id}_{j}",
            "customer_id": cust_id,
            "sender_account": sender_acc,
            "receiver_account": f"CRYPTO_EXCH_{j:02d}",
            "amount": 14500.0,
            "currency": "USD",
            "payment_format": "Wire",
            "is_laundering": 1
        })

    # --- 4. Pattern 3: Massive Amount Anomaly (Customer 8802) ---
    cust_id = "8802"
    base_time = start_date + timedelta(days=25)
    records.append({
        "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id": f"TXN_SPIKE_{cust_id}",
        "customer_id": cust_id,
        "sender_account": f"ACC_{cust_id}_01",
        "receiver_account": "UNKNOWN_PRIVATE_ACC",
        "amount": 495000.0,
        "currency": "USD",
        "payment_format": "Wire",
        "is_laundering": 1
    })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("dt").drop(columns=["dt"]).reset_index(drop=True)
    
    return df


def save_dataset(output_path: str = "data/transactions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_aml_data()
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset successfully created at: {output_path}")
    print(f"    Total rows: {len(df)}")
    print(f"    Laundering cases: {df['is_laundering'].sum()} ({df['is_laundering'].mean()*100:.2f}%)")
    print(f"    Unique customers: {df['customer_id'].nunique()}")


if __name__ == "__main__":
    save_dataset()
