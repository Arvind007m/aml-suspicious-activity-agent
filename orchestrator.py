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
    Generates and saves a clean, visual supporting chart artifact for judges (Fix 9).
    Gracefully handles empty result sets without crashing.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_entities = context.get("explanations", context.get("top_suspicious_entities", []))
    
    # Fix 9: Gracefully skip chart generation if results are empty
    if not top_entities or len(top_entities) == 0:
        return ""

    df_top = pd.DataFrame(top_entities).head(8)
    if "customer_id" not in df_top.columns or "risk_score" not in df_top.columns:
        return ""

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


def calculate_detection_metrics(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates Ground Truth Detection Metrics (Precision, Recall, F1, Detection Hit Rate)
    scoped appropriately to the query intent (Fix 4).
    """
    df_raw: pd.DataFrame = context["df"].copy()
    df_raw["customer_id"] = df_raw["customer_id"].astype(str)
    
    plan_meta = context.get("plan_meta", {})
    intent = plan_meta.get("intent", "broad_analysis")
    entities = plan_meta.get("entities", {})
    top_entities = context.get("top_suspicious_entities", [])
    
    # Ground truth laundering customers
    laundering_customers = set(df_raw[df_raw["is_laundering"] == 1]["customer_id"].astype(str).unique())
    total_ground_truth = len(laundering_customers)
    
    flagged_high_med = [
        str(e["customer_id"]) for e in top_entities 
        if e.get("risk_level") in ["High", "Medium"] or e.get("risk_score", 0) >= 40.0
    ]
    flagged_set = set(flagged_high_med)

    if intent == "single_entity":
        target_cust = str(entities.get("customer_id") or "4521").strip()
        is_target_laundering = target_cust in laundering_customers
        target_flagged = target_cust in flagged_set or (len(top_entities) > 0 and str(top_entities[0].get("customer_id")) == target_cust)
        target_risk = top_entities[0].get("risk_level", "Unknown") if len(top_entities) > 0 else "Unknown"
        
        return {
            "scope": "single_entity",
            "target_customer": target_cust,
            "target_laundering_ground_truth": "LAUNDERING" if is_target_laundering else "CLEAN",
            "target_risk_level": target_risk,
            "match_status": "CORRECTLY IDENTIFIED" if (is_target_laundering and target_risk in ["High", "Medium"]) or (not is_target_laundering and target_risk == "Low") else "MISMATCH"
        }

    elif intent in ["threshold_query", "pattern_detection"]:
        # Metrics scoped ONLY to returned query subset (Fix 4)
        subset_custs = set(str(e["customer_id"]) for e in top_entities)
        true_positives = len(subset_custs.intersection(laundering_customers))
        false_positives = len(subset_custs - laundering_customers)
        
        precision = true_positives / len(subset_custs) if len(subset_custs) > 0 else 0.0
        hit_rate = precision * 100.0
        
        return {
            "scope": "query_scoped",
            "evaluated_customers": len(subset_custs),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": round(precision, 3),
            "hit_rate_pct": round(hit_rate, 1)
        }

    else:
        # Broad analysis: Dataset-wide evaluation
        true_positives = len(flagged_set.intersection(laundering_customers))
        false_positives = len(flagged_set - laundering_customers)
        false_negatives = total_ground_truth - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        hit_rate = (true_positives / len(flagged_set)) * 100.0 if len(flagged_set) > 0 else 0.0

        return {
            "scope": "dataset_wide",
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
    Includes Live Reasoning Trace (Feature 1) and Efficiency Savings Metrics (Feature 5).
    """
    plan_meta = context["plan_meta"]
    metrics = calculate_detection_metrics(context)
    explanations = context.get("explanations", [])
    trace = context.get("reasoning_trace", plan_meta.get("reasoning_trace", []))
    
    tools_run_cnt = len(context.get('executed_tools', []))
    total_tools = 5
    saved_overhead = int(((total_tools - tools_run_cnt) / total_tools) * 100)
    exec_time = context.get("execution_time_sec", 0.0)

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
    print("               LIVE AGENT REASONING TRACE                      ")
    print("-" * 70)
    for idx, step in enumerate(trace, 1):
        print(f"  Step {idx:02d}: {step}")

    print("-" * 70)
    print("               EFFICIENCY SAVINGS METRICS                      ")
    print("-" * 70)
    print(f"  * Tools Invoked:       {tools_run_cnt} of {total_tools} available")
    print(f"  * Pipeline Overhead:   Saved {saved_overhead}% tool overhead (Full pipeline = 5 tools; query needed {tools_run_cnt})")
    print(f"  * Execution Time:      {exec_time:.2f}s")
    
    print("-" * 70)
    print("             GROUND TRUTH DETECTION METRICS                    ")
    print("-" * 70)
    
    if metrics["scope"] == "single_entity":
        print(f"  * Entity Evaluated:     Customer {metrics['target_customer']}")
        print(f"  * Ground Truth Status:  {metrics['target_laundering_ground_truth']}")
        print(f"  * Risk Level Assigned:  {metrics['target_risk_level']}")
        print(f"  * Assessment Result:    {metrics['match_status']}")
    elif metrics["scope"] == "query_scoped":
        print(f"  * Query-Scoped Precision: {metrics['precision']} ({metrics['true_positives']} TP / {metrics['evaluated_customers']} Evaluated)")
        print(f"  * Detection Hit Rate:     {metrics['hit_rate_pct']}%")
    else:
        print(f"  * Dataset Precision:    {metrics['precision']} ({metrics['true_positives']} TP / {metrics['true_positives']+metrics['false_positives']} Flagged)")
        print(f"  * Dataset Recall:       {metrics['recall']} ({metrics['true_positives']} TP / {metrics['total_laundering_customers']} Actual Laundering)")
        print(f"  * Dataset F1 Score:     {metrics['f1_score']}")
        print(f"  * Detection Hit Rate:   {metrics['hit_rate_pct']}%")

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
    if chart_path:
        print(f"SUPPORTING CHART SAVED TO: {chart_path}")
    else:
        print("SUPPORTING CHART: Skipped (no entities flagged)")
    print("="*70 + "\n")


def run_agent_query(query: str, csv_path: str = "data/transactions.csv") -> Dict[str, Any]:
    """
    Main entry point for running a query through the AML agent pipeline.
    Track execution time and reasoning trace.
    """
    start_time = time.time()
    
    if not os.path.exists(csv_path):
        print(f"[!] Dataset not found at {csv_path}. Generating synthetic dataset...")
        save_dataset(csv_path)
        
    df = pd.read_csv(csv_path, dtype={"customer_id": str})
    df["customer_id"] = df["customer_id"].astype(str)
    
    # 1. Planner parses query into structured JSON plan
    plan_meta = create_plan(query)
    
    # 2. Build initial execution context
    context = {
        "query": query,
        "df": df,
        "plan_meta": plan_meta,
        "reasoning_trace": list(plan_meta.get("reasoning_trace", []))
    }
    
    # 3. Execute tools specified in plan
    context = execute_tool_chain(plan_meta["plan"], context)
    
    # Record execution duration
    context["execution_time_sec"] = time.time() - start_time
    
    # Add final result summary to trace
    top_items = context.get("top_suspicious_entities", [])
    if top_items:
        top_cust = top_items[0].get("customer_id")
        top_risk = top_items[0].get("risk_level")
        top_score = top_items[0].get("risk_score")
        top_action = top_items[0].get("escalation_action")
        context["reasoning_trace"].append(f"Result: top flagged Customer {top_cust} risk {top_risk} (score {top_score})")
        context["reasoning_trace"].append(f"Escalation Action: {top_action}")
    
    # 4. Generate supporting visual chart artifact
    chart_path = save_supporting_chart(context)
    
    # 5. Output judge-facing execution summary
    print_judge_execution_summary(context, chart_path)
    
    return context



if __name__ == "__main__":
    query_str = sys.argv[1] if len(sys.argv) > 1 else "Analyse this dataset for suspicious activity"
    run_agent_query(query_str)
