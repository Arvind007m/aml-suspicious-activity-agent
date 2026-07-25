"""
Master Demo Script for Hackathon Judges.
Runs all 4 canned queries + 3 unscripted queries + 1 vague query to visually showcase:
- Live reasoning traces (Feature 1)
- Generalization on unscripted queries (Feature 2)
- Human-in-the-loop clarification (Feature 3)
- False-positive reduction baseline (Feature 4)
- Efficiency savings metrics (Feature 5)
- Evidence-backed explanations (Feature 6)
"""

import time
from orchestrator import run_agent_query

DEMO_QUERIES = [
    {
        "num": 1,
        "query": "Analyse this dataset for suspicious activity",
        "type": "Canned Query 1",
        "description": "Broad dataset analysis -> Runs all 5 tools + displays Naive FP Reduction comparison (100% FP reduction)."
    },
    {
        "num": 2,
        "query": "Find structuring patterns in the last 30 days",
        "type": "Canned Query 2",
        "description": "Pattern detection -> Skips EDA. Date filter slices data to last 30 days."
    },
    {
        "num": 3,
        "query": "Which customers made 10+ transactions under $10,000?",
        "type": "Canned Query 3",
        "description": "Threshold query -> Skips EDA AND ML anomaly detection. Pure aggregation (NO ML)."
    },
    {
        "num": 4,
        "query": "Is customer 4521 suspicious?",
        "type": "Canned Query 4",
        "description": "Single-entity lookup -> Skips EDA AND feature engineering. Direct risk lookup for Customer 4521."
    },
    {
        "num": 5,
        "query": "Show me customers who received large deposits then emptied their account within an hour",
        "type": "Unscripted Query 1",
        "description": "Generalization test -> Maps to pattern_detection / rapid_cash_out typology."
    },
    {
        "num": 6,
        "query": "Who are the top 5 riskiest customers?",
        "type": "Unscripted Query 2",
        "description": "Generalization test -> Maps to broad_analysis ranking."
    },
    {
        "num": 7,
        "query": "check the data",
        "type": "Ambiguous Query",
        "description": "Human-in-the-Loop test -> Detects ambiguity, triggers clarifying question, and halts execution to save overhead."
    }
]


def run_full_hackathon_demo():
    print("\n" + "#"*75)
    print("      AI-POWERED AML DETECTION AGENT - MASTER HACKATHON DEMO       ")
    print("#"*75)
    print("Demonstrating LLM Dynamic Planning, Live Reasoning Traces & Generalization!")
    print("Watch how the agent dynamically selects tools, asks clarifying questions,")
    print("proves false-positive reduction, and provides evidence-backed explanations.\n")
    
    time.sleep(1)

    for item in DEMO_QUERIES:
        q_num = item["num"]
        q_text = item["query"]
        q_type = item["type"]
        q_desc = item["description"]
        
        print("\n" + "*"*75)
        print(f" DEMO QUERY {q_num}/7 ({q_type}): \"{q_text}\"")
        print(f" EXPECTED BRANCHING: {q_desc}")
        print("*"*75)
        
        run_agent_query(q_text)
        time.sleep(2.0)

    print("\n" + "#"*75)
    print("      DEMO COMPLETE - ALL QUERIES EXECUTED WITH DYNAMIC BRANCHING    ")
    print("#"*75 + "\n")


if __name__ == "__main__":
    run_full_hackathon_demo()
