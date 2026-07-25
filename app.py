"""
Streamlit UI Dashboard for AML Agent.
Interactive interface for judges to run queries and observe dynamic planning & tool execution.
"""

import os
import pandas as pd
import streamlit as st
from orchestrator import run_agent_query

st.set_page_config(
    page_title="AI AML Agent - Dynamic Planner",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Powered AML Suspicious Activity Agent")
st.caption("Powered by Groq LLM Planner (`llama-3.3-70b-versatile`) & Dynamic Tool Execution")

st.markdown("""
This agent dynamically inspects queries, generates a **Structured Execution Plan**, 
selects appropriate tools, and executes only the required analysis components.
""")

# Pre-defined sample queries
sample_queries = [
    "Analyse this dataset for suspicious activity",
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is customer 4521 suspicious?"
]

selected_query = st.selectbox("Select Sample Query or Type Custom Query:", sample_queries)
custom_query = st.text_input("Or Enter Custom Query:", value=selected_query)

if st.button("🚀 Run Agent Analysis", type="primary"):
    with st.spinner("Agent planning and executing tool chain..."):
        context = run_agent_query(custom_query)
        plan_meta = context["plan_meta"]
        
        st.success("Execution Complete!")
        
        # --- Planner & Execution Overview ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Planner Source", plan_meta.get("planner_type", "Rule Engine"))
        col2.metric("Intent", plan_meta.get("intent"))
        col3.metric("Tools Executed", ", ".join(context.get("executed_tools", [])))

        st.subheader("📋 Agent Plan Details")
        st.json({
            "intent": plan_meta.get("intent"),
            "filters": plan_meta.get("filters"),
            "aml_pattern": plan_meta.get("aml_pattern"),
            "plan": plan_meta.get("plan"),
            "skipped": plan_meta.get("skipped"),
            "reason": plan_meta.get("reason")
        })

        # --- Chart Artifact ---
        latest_chart = "charts/latest_analysis.png"
        if os.path.exists(latest_chart):
            st.subheader("📊 Supporting Visual Chart")
            st.image(latest_chart, use_container_width=True)

        # --- Top Flagged Entities ---
        st.subheader("🚨 Top Flagged Suspicious Entities")
        explanations = context.get("explanations", [])
        if explanations:
            df_display = pd.DataFrame(explanations)[[
                "customer_id", "risk_level", "risk_score", "ground_truth_laundering", "explanation", "escalation_action"
            ]]
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No suspicious entities flagged for this query.")
