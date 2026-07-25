"""
Master Demo Script for Hackathon Judges.
Runs all 4 example queries in sequence to visually showcase dynamic agentic branching,
tool skipping, ground truth metrics, and execution summaries.
"""

import time
from orchestrator import run_agent_query

DEMO_QUERIES = [
    {
        "num": 1,
        "query": "Analyse this dataset for suspicious activity",
        "description": "Broad dataset analysis -> Runs all tools (EDA, features, IsolationForest ML, risk scoring, explanations)."
    },
    {
        "num": 2,
        "query": "Find structuring patterns in the last 30 days",
        "description": "Pattern detection -> Skips EDA. Focuses on velocity/smurfing features, date filter, ML anomaly scoring."
    },
    {
        "num": 3,
        "query": "Which customers made 10+ transactions under $10,000?",
        "description": "Threshold query -> Skips EDA AND ML anomaly detection. Pure aggregation & threshold filtering (NO ML)."
    },
    {
        "num": 4,
        "query": "Is customer 4521 suspicious?",
        "description": "Single-entity lookup -> Skips EDA AND feature engineering. Direct risk lookup for Customer 4521."
    }
]


def run_full_hackathon_demo():
    print("\n" + "#"*75)
    print("      AI-POWERED AML DETECTION AGENT - HACKATHON DEMO EXECUTION       ")
    print("#"*75)
    print("Demonstrating dynamic agentic planning (Groq LLM / Rule Fallback)")
    print("Watch how the agent dynamically selects and skips tools based on query intent!\n")
    
    time.sleep(1)

    for item in DEMO_QUERIES:
        q_num = item["num"]
        q_text = item["query"]
        q_desc = item["description"]
        
        print("\n" + "*"*75)
        print(f" DEMO QUERY {q_num}/4: \"{q_text}\"")
        print(f" EXPECTED BRANCHING: {q_desc}")
        print("*"*75)
        
        run_agent_query(q_text)
        # Fix 8: Add 2-second rate-limit sleep between queries
        time.sleep(2.0)

    print("\n" + "#"*75)
    print("      DEMO COMPLETE - ALL 4 QUERIES EXECUTED WITH DYNAMIC BRANCHING    ")
    print("#"*75 + "\n")


if __name__ == "__main__":
    run_full_hackathon_demo()
