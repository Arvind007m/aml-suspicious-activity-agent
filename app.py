"""
Streamlit UI Dashboard for AML Agent.
Interactive presentation interface designed for hackathon judge clarity and maximum demo impact.
"""

import os
import pandas as pd
import streamlit as st
from orchestrator import run_agent_query, calculate_detection_metrics
from tools.graph_viz import build_flow_dot


st.set_page_config(
    page_title="AI AML Agent - Dynamic Planner",
    layout="wide"
)

# Custom CSS for high-impact judge presentation styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .decision-box {
        background-color: #0F172A;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered AML Suspicious Activity Agent")
st.caption("Powered by Groq LLM Planner (`llama-3.3-70b-versatile`) & Dynamic Tool Execution")

st.markdown("""
This agent dynamically inspects queries, generates a **Structured Execution Plan**, 
selects appropriate tools, and executes only the required analysis components.
""")

# Pre-defined sample queries
sample_queries = [
    "Analyse this dataset for suspicious activity",
    "Show me customers who received large deposits then emptied their account within an hour",
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is customer 4521 suspicious?",
    "check the data"
]

selected_query = st.selectbox("Select Sample Query or Type Custom Query:", sample_queries)
custom_query = st.text_input("Or Enter Custom Query:", value=selected_query)

if st.button("🚀 Run Agent Analysis", type="primary"):
    with st.spinner("Agent planning and executing tools..."):
        context = run_agent_query(custom_query)
        plan_meta = context["plan_meta"]
        
        # --- Feature 3: Human-in-the-Loop Clarification UI ---
        if plan_meta.get("intent") == "needs_clarification":
            st.warning("❓ **Human-in-the-Loop Clarification Required**")
            st.info(f"**Clarifying Question**: {plan_meta.get('clarifying_question')}")
            st.caption("The agent halted tool execution to save overhead because the query was too ambiguous.")
            
            st.subheader("🧠 Agent Reasoning Trace")
            for idx, step in enumerate(plan_meta.get("reasoning_trace", []), 1):
                st.write(f"**Step {idx:02d}**: {step}")

        else:
            st.success("Execution Complete!")
            metrics = calculate_detection_metrics(context)
            explanations = context.get("explanations", [])
            trace = context.get("reasoning_trace", plan_meta.get("reasoning_trace", []))
            intent = plan_meta.get("intent", "broad_analysis")
            aml_pattern = plan_meta.get("aml_pattern")
            exec_tools = context.get("executed_tools", [])
            skipped_tools = plan_meta.get("skipped", [])
            exec_time = context.get("execution_time_sec", 0.0)

            # Count High risk entities
            high_risk_cnt = sum(1 for e in explanations if e.get("risk_level") == "High")
            total_flagged = len(explanations)

            # --- 1. Headline Metrics Row at Top (Fix 1) ---
            st.subheader("📊 Execution Headline Metrics")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            m_col1.metric("Planner Source", plan_meta.get("planner_type", "LLM Planner").replace("Groq LLM (llama-3.3-70b-versatile)", "Groq LLM"))
            m_col2.metric("Intent & Pattern", f"{intent}", delta=f"Pattern: {aml_pattern}" if aml_pattern else None)
            m_col3.metric("Flagged Risk", f"{high_risk_cnt} High Risk", delta=f"{total_flagged} total flagged")

            if intent == "broad_analysis":
                # PROMINENT FP REDUCTION metric card for broad analysis
                fp_red = metrics.get("fp_reduction_pct", 100.0)
                naive_fp = metrics.get("naive_false_positives", 76)
                agent_fp = metrics.get("false_positives", 0)
                m_col4.metric("False Positives Reduced", f"{fp_red:.1f}%", delta=f"{agent_fp} agent vs {naive_fp} naive FP", delta_color="normal")
                m_col5.metric("Tool Savings", "Full Analysis (5/5 tools)", delta=f"Execution: {exec_time:.2f}s")
            else:
                tools_run_cnt = len(exec_tools)
                saved_overhead = int(((5 - tools_run_cnt) / 5) * 100)
                m_col4.metric("Tool Savings", f"{saved_overhead}% Saved", delta=f"{tools_run_cnt}/5 tools run")
                m_col5.metric("Execution Time", f"{exec_time:.2f}s")

            # --- 2. Clear Agent Decision Summary Panel (Fix 2) ---
            st.markdown("### 📋 Agent Decision Summary")
            with st.container():
                st.markdown(f"""
                <div class="decision-box">
                    <p style="margin-bottom:8px;"><b>QUERY:</b> <code>"{context['query']}"</code></p>
                    <p style="margin-bottom:8px;"><b>DETECTED INTENT:</b> <code>{intent}</code> | <b>AML PATTERN:</b> <code>{aml_pattern or 'None'}</code></p>
                    <p style="margin-bottom:8px;"><b>FILTERS & ENTITIES:</b> Customer ID: <code>{plan_meta.get('entities', {}).get('customer_id') or 'None'}</code> | Date Filter: <code>{plan_meta.get('filters', {}).get('date_range_days') or 'Full Dataset'}</code></p>
                    <p style="margin-bottom:8px;"><b>PROPOSED PLAN:</b> <code>{plan_meta.get('plan')}</code></p>
                    <p style="margin-bottom:0px;"><b>SKIPPED TOOLS & RATIONALE:</b> <code>{skipped_tools}</code> — <i>{plan_meta.get('reason')}</i></p>
                </div>
                """, unsafe_allow_html=True)

            # --- 3. Agent Execution Flow Visual Diagram ---
            st.subheader("🌐 Agent Execution Flow")
            res_summary_txt = f"{high_risk_cnt} High Risk Flagged" if high_risk_cnt > 0 else "Analysis Complete"
            flow_dot_code = build_flow_dot(plan_meta, result_summary=res_summary_txt)
            st.graphviz_chart(flow_dot_code)

            # --- 4. Compress & Color Reasoning Trace ---
            st.subheader("🧠 Live Agent Reasoning Trace")

            
            # Separate decision steps from verbose tool execution logs
            decision_steps = [s for s in trace if not s.startswith("Step") or "Running" not in s]
            log_steps = [s for s in trace if "Running" in s]

            for step in decision_steps:
                if "SKIP" in step:
                    st.warning(f"**Decision**: {step}")
                elif "Result" in step or "Escalation" in step:
                    st.success(f"**Outcome**: {step}")
                else:
                    st.info(f"**Trace Step**: {step}")

            if log_steps:
                with st.expander("🛠️ View Detailed Tool Execution Logs"):
                    for log_s in log_steps:
                        st.code(log_s, language="bash")

            with st.expander("📋 View Raw Agent Plan JSON"):
                st.json({
                    "intent": plan_meta.get("intent"),
                    "filters": plan_meta.get("filters"),
                    "aml_pattern": plan_meta.get("aml_pattern"),
                    "plan": plan_meta.get("plan"),
                    "skipped": plan_meta.get("skipped"),
                    "reason": plan_meta.get("reason")
                })

            st.divider()

            # --- 4. Grouped Requirements Output Section (Fix 4) ---
            st.markdown("## 🎯 Agent Output (per requirements)")

            # --- 5. Detection Metrics Panel (Fix 5) ---
            st.markdown("### 1. Detection Performance & Baseline Metrics")
            if metrics["scope"] == "broad_analysis" or metrics["scope"] == "dataset_wide":
                p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
                p_col1.metric("Dataset Precision", f"{metrics.get('precision', 1.0):.3f}")
                p_col2.metric("Dataset Recall", f"{metrics.get('recall', 0.75):.3f}")
                p_col3.metric("F1 Score", f"{metrics.get('f1_score', 0.857):.3f}")
                p_col4.metric("Naive FP Baseline", f"{metrics.get('naive_false_positives', 76)} False Positives", delta=f"{metrics.get('naive_flagged_cnt', 80)} total flagged")
                p_col5.metric("Agent Impact", f"{metrics.get('fp_reduction_pct', 100.0):.1f}% FP Reduction", delta=f"{metrics.get('false_positives', 0)} agent FP")
            elif metrics["scope"] == "single_entity":
                st.info(f"🎯 **Single Entity Lookup**: Customer **{metrics.get('target_customer')}** evaluated | Ground Truth: **{metrics.get('target_laundering_ground_truth')}** | Assigned Risk: **{metrics.get('target_risk_level')}**")
            else:
                st.info(f"🎯 **Query-Scoped Evaluation**: Precision **{metrics.get('precision')}** ({metrics.get('true_positives')} TP / {metrics.get('evaluated_customers')} Evaluated) | Hit Rate: **{metrics.get('hit_rate_pct')}%**")

            # --- 6. Top Suspicious Entities Table (Fix 6: Highlighting & Text Wrapping) ---
            st.markdown("### 2. Top Suspicious Entities & Evidence Explanations")
            if explanations:
                target_cols = [
                    "customer_id", "risk_level", "confidence", "risk_score", 
                    "ground_truth_laundering", "explanation", "escalation_action"
                ]
                df_display = pd.DataFrame(explanations)
                valid_cols = [c for c in target_cols if c in df_display.columns]
                df_display = df_display[valid_cols]

                # Style High Risk rows with red tint highlighting
                def highlight_high_risk(row):
                    if row.get("risk_level") == "High":
                        return ["background-color: #451A1A; color: #FCA5A5; font-weight: bold"] * len(row)
                    elif row.get("risk_level") == "Medium":
                        return ["background-color: #332615; color: #FDE047"] * len(row)
                    else:
                        return [""] * len(row)

                styled_df = df_display.style.apply(highlight_high_risk, axis=1)

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    column_config={
                        "customer_id": st.column_config.TextColumn("Customer ID", width="small"),
                        "risk_level": st.column_config.TextColumn("Risk Level", width="small"),
                        "confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "risk_score": st.column_config.NumberColumn("Risk Score", format="%.1f", width="small"),
                        "ground_truth_laundering": st.column_config.NumberColumn("Ground Truth", width="small"),
                        "explanation": st.column_config.TextColumn("Evidence-Backed Explanation", width="large"),
                        "escalation_action": st.column_config.TextColumn("Escalation Recommendation", width="medium")
                    }
                )
            else:
                st.info("No suspicious entities flagged for this query.")

            # --- 7. Supporting Visual Artifacts ---
            st.markdown("### 3. Supporting Visual Artifacts & Network Topology")
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
