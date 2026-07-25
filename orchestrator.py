"""
Orchestrator: CLI Entry Point for AML Agent.
Takes user queries, calls LLM planner, executes tool chain, computes detection metrics,
prints judge-facing execution summaries, and saves chart artifacts.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from planner import create_plan
from registry import execute_tool_chain
from generate_data import generate_synthetic_aml_data, save_dataset


def save_supporting_chart(context: Dict[str, Any], output_dir: str = "charts") -> str:
    """
    Generates and saves a clean, visual supporting chart artifact for judges.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_entities = context.get("explanations", context.get("top_suspicious_entities", []))
    
    if not top_entities:
        return ""

    df_top = pd.DataFrame(top_entities).head(8)
    
    plt.figure(figsize=(10, 5))
    colors = []
    for level in df_top.get("risk_level", ["Low"] * len(df_top)):
        if level == "High":
            colors.append("#d9534f")  # Red
        elif level == "Medium":
            colors.append("#f0ad4e")  # Orange
        else:
            colors.append("#5cb85c")  # Green
            
    cust_labels = [f"Cust {c}" for c in df_top["customer_id"]]
    risk_scores = df_top.get("risk_score", [50] * len(df_top))
    
    bars = plt.bar(cust_labels, risk_scores, color=colors, edgecolor="black", alpha=0.85)
    
    plt.title("AML Agent Risk Assessment - Top Flagged Entities", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Customer ID", fontsize=11, fontweight="bold")
    plt.ylabel("Risk Score (0 - 100)", fontsize=11, fontweight="bold")
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Value labels on top of bars
    for bar, score, level in zip(bars, risk_scores, df_top.get("risk_level", [])):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{score:.0f}\n({level})", ha='center', va='bottom', fontsize=9, fontweight='bold')

    timestamp_str = int(time.time())
    chart_filename = f"chart_{timestamp_str}.png"
    filepath = os.path.join(output_dir, chart_filename)
    latest_filepath = os.path.join(output_dir, "latest_analysis.png")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.savefig(latest_filepath, dpi=150)
    plt.close()
    
    return filepath


def calculate_detection_metrics(context: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates Ground Truth Detection Metrics (Precision, Recall, F1, Detection Hit Rate)
    using is_laundering ground truth labels.
    """
    df_raw: pd.DataFrame = context["df"]
    top_entities = context.get("top_suspicious_entities", [])
    
    # Ground truth laundering customers
    laundering_customers = set(df_raw[df_raw["is_laundering"] == 1]["customer_id"].astype(str).unique())
    total_ground_truth = len(laundering_customers)
    
    flagged_high_med = [
        str(e["customer_id"]) for e in top_entities 
        if e.get("risk_level") in ["High", "Medium"] or e.get("risk_score", 0) >= 40.0
    ]
    
    flagged_set = set(flagged_high_med)
    true_positives = len(flagged_set.intersection(laundering_customers))
    false_positives = len(flagged_set - laundering_customers)
    false_negatives = total_ground_truth - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hit_rate = (true_positives / len(flagged_set)) * 100.0 if len(flagged_set) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "hit_rate_pct": round(hit_rate, 1),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "total_laundering_customers": total_ground_truth
    }


def print_judge_execution_summary(context: Dict[str, Any], chart_path: str):
    """
    Prints a clean, judge-facing execution summary detailing plan, execution, metrics, and explanations.
    """
    plan_meta = context["plan_meta"]
    metrics = calculate_detection_metrics(context)
    explanations = context.get("explanations", [])
    
    print("\n" + "="*70)
    print("                JUDGE-FACING EXECUTION SUMMARY                 ")
    print("="*70)
    print(f"QUERY:         \"{context['query']}\"")
    print(f"PLANNER TYPE:  {plan_meta.get('planner_type', 'LLM Planner')}")
    print(f"INTENT:        {plan_meta.get('intent')}")
    print(f"FILTERS:       {plan_meta.get('filters')}")
    print(f"AML PATTERN:   {plan_meta.get('aml_pattern')}")
    print(f"PLAN:          {plan_meta.get('plan')}")
    print(f"SKIPPED:       {plan_meta.get('skipped')}")
    print(f"TOOLS RUN:     {context.get('executed_tools')}")
    print(f"PLAN REASON:   {plan_meta.get('reason')}")
    
    print("-" * 70)
    print("             GROUND TRUTH DETECTION METRICS                    ")
    print("-" * 70)
    print(f"  * Precision:           {metrics['precision']} ({metrics['true_positives']} TP / {metrics['true_positives']+metrics['false_positives']} Flagged)")
    print(f"  * Recall:              {metrics['recall']} ({metrics['true_positives']} TP / {metrics['total_laundering_customers']} Actual Laundering)")
    print(f"  * F1 Score:            {metrics['f1_score']}")
    print(f"  * Detection Hit Rate:  {metrics['hit_rate_pct']}%")

    print("-" * 70)
    print("             TOP SUSPICIOUS ENTITIES & EXPLANATIONS             ")
    print("-" * 70)
    
    if not explanations:
        print("  No suspicious entities flagged for this query.")
    else:
        for idx, item in enumerate(explanations[:5], 1):
            cust_id = item.get("customer_id")
            risk = item.get("risk_level", "Low")
            score = item.get("risk_score", 0.0)
            ground_truth = "LAUNDERING" if item.get("ground_truth_laundering", 0) > 0 else "CLEAN"
            expl = item.get("explanation", "N/A")
            action = item.get("escalation_action", "N/A")
            
            print(f"  [{idx}] Customer ID: {cust_id} | Risk: {risk} (Score: {score}) | Ground Truth: {ground_truth}")
            print(f"      Explanation: {expl}")
            print(f"      Action:      {action}\n")

    print("-" * 70)
    print(f"SUPPORTING CHART SAVED TO: {chart_path}")
    print("="*70 + "\n")


def run_agent_query(query: str, csv_path: str = "data/transactions.csv") -> Dict[str, Any]:
    """
    Main entry point for running a query through the AML agent pipeline.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic dataset...")
        save_dataset(csv_path)
        
    df = pd.read_csv(csv_path)
    
    # 1. Planner parses query into structured JSON plan
    plan_meta = create_plan(query)
    
    # 2. Build initial execution context
    context = {
        "query": query,
        "df": df,
        "plan_meta": plan_meta
    }
    
    # 3. Execute tools specified in plan
    context = execute_tool_chain(plan_meta["plan"], context)
    
    # 4. Generate supporting visual chart artifact
    chart_path = save_supporting_chart(context)
    
    # 5. Output judge-facing execution summary
    print_judge_execution_summary(context, chart_path)
    
    return context


if __name__ == "__main__":
    query_str = sys.argv[1] if len(sys.argv) > 1 else "Analyse this dataset for suspicious activity"
    run_agent_query(query_str)
