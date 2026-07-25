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
    "Is customer 4521 suspicious?",
    "check the data"
]

selected_query = st.selectbox("Select Sample Query or Type Custom Query:", sample_queries)
custom_query = st.text_input("Or Enter Custom Query:", value=selected_query)

if st.button("🚀 Run Agent Analysis", type="primary"):
    with st.spinner("Agent planning..."):
        context = run_agent_query(custom_query)
        plan_meta = context["plan_meta"]
        
        # Feature 3: Human-in-the-Loop Clarification UI
        if plan_meta.get("intent") == "needs_clarification":
            st.warning("❓ **Human-in-the-Loop Clarification Required**")
            st.write(f"**Clarifying Question**: {plan_meta.get('clarifying_question')}")
            st.info("The agent halted tool execution to save overhead because the query was too ambiguous.")
            
            st.subheader("🧠 Agent Reasoning Trace")
            for idx, step in enumerate(plan_meta.get("reasoning_trace", []), 1):
                st.write(f"**Step {idx:02d}**: {step}")
        else:
            st.success("Execution Complete!")
            
            # --- Planner & Execution Overview ---
            tools_run_cnt = len(context.get('executed_tools', []))
            total_tools = 5
            saved_overhead = int(((total_tools - tools_run_cnt) / total_tools) * 100)
            exec_time = context.get("execution_time_sec", 0.0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Planner Source", plan_meta.get("planner_type", "Rule Engine"))
            col2.metric("Intent", plan_meta.get("intent"))
            col3.metric("Tool Savings", f"{saved_overhead}%", delta=f"{tools_run_cnt}/5 tools run")
            col4.metric("Execution Time", f"{exec_time:.2f}s")

            # --- Live Reasoning Trace (Feature 1) ---
            st.subheader("🧠 Live Agent Reasoning Trace")
            trace = context.get("reasoning_trace", plan_meta.get("reasoning_trace", []))
            for idx, step in enumerate(trace, 1):
                if "SKIP" in step:
                    st.warning(f"**Step {idx:02d}**: {step}")
                elif "Result" in step or "Escalation" in step:
                    st.success(f"**Step {idx:02d}**: {step}")
                else:
                    st.info(f"**Step {idx:02d}**: {step}")

            # --- Plan JSON Details ---
            with st.expander("📋 View Raw Agent Plan JSON"):
                st.json({
                    "intent": plan_meta.get("intent"),
                    "filters": plan_meta.get("filters"),
                    "aml_pattern": plan_meta.get("aml_pattern"),
                    "plan": plan_meta.get("plan"),
                    "skipped": plan_meta.get("skipped"),
                    "reason": plan_meta.get("reason")
                })

            # --- Supporting Charts & Network Topology ---
            latest_chart = "charts/latest_analysis.png"
            latest_network = "charts/latest_network.png"

            col_chart, col_net = st.columns(2)
            with col_chart:
                if os.path.exists(latest_chart):
                    st.subheader("📊 Risk Profile Chart")
                    st.image(latest_chart, use_container_width=True)

            with col_net:
                if os.path.exists(latest_network):
                    st.subheader("🌐 Transaction Network Topology")
                    st.image(latest_network, use_container_width=True)
                else:
                    st.info("No transaction network graph generated for this query.")

            # --- Top Flagged Entities ---
            st.subheader("🚨 Top Flagged Suspicious Entities")
            explanations = context.get("explanations", [])
            if explanations:
                target_cols = [
                    "customer_id", "risk_level", "risk_score", "ground_truth_laundering", "explanation", "escalation_action"
                ]
                df_display = pd.DataFrame(explanations)
                valid_cols = [c for c in target_cols if c in df_display.columns]
                df_display = df_display[valid_cols]
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No suspicious entities flagged for this query.")

