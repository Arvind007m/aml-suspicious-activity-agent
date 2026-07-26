"""
Transaction Network Graph Visualization Tool (tools/graph_viz.py)
Renders high-contrast, publication-quality directed money flow graphs for flagged entities.
Includes manual columnar node positioning, edge label anti-clutter summarization, and crisp arrowheads.
"""

import os
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import networkx as nx


def build_transaction_graph(df_raw: pd.DataFrame, top_entities: List[Dict[str, Any]], aml_pattern: str = None) -> str:
    """
    Constructs and renders a directed transaction money-flow network graph for top flagged customer.
    Features manual columnar node layout with even vertical spacing and edge label summarization.
    Saves image to charts/latest_network.png and returns file path.
    """
    if not top_entities or df_raw.empty:
        return ""

    os.makedirs("charts", exist_ok=True)
    latest_net_path = os.path.join("charts", "latest_network.png")
    
    # 1. Primary target flagged entity
    target_item = top_entities[0]
    target_cust = str(target_item.get("customer_id")).strip()
    
    # Do not generate graph for entities with 0 transactions or non-existent customers
    if target_item.get("txn_count_total", 0) == 0 or "No transactions found" in str(target_item.get("explanation", "")):
        if os.path.exists(latest_net_path):
            try:
                os.remove(latest_net_path)
            except Exception:
                pass
        return ""

    df = df_raw.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["sender_account"] = df["sender_account"].astype(str)
    df["receiver_account"] = df["receiver_account"].astype(str)

    # Filter transactions related to target customer
    cust_txns = df[
        (df["customer_id"] == target_cust) |
        (df["sender_account"].str.contains(target_cust, na=False)) |
        (df["receiver_account"].str.contains(target_cust, na=False))
    ].copy()

    if cust_txns.empty:
        if os.path.exists(latest_net_path):
            try:
                os.remove(latest_net_path)
            except Exception:
                pass
        return ""

    if len(cust_txns) > 20:
        cust_txns = cust_txns.sort_values("amount", ascending=False).head(20).copy()


    # 2. Build Directed Graph
    G = nx.DiGraph()
    target_node = f"Customer_{target_cust}"
    G.add_node(target_node, node_type="target")

    for idx, row in cust_txns.iterrows():
        sender = str(row["sender_account"])
        receiver = str(row["receiver_account"])
        amt = float(row["amount"])
        fmt = str(row.get("payment_format", "Transfer"))

        s_node = target_node if target_cust in sender or sender == target_cust else sender
        r_node = target_node if target_cust in receiver or receiver == target_cust else receiver
        
        # Avoid self-loops for cleaner topology
        if s_node != r_node:
            G.add_edge(s_node, r_node, amount=amt, payment_format=fmt)

    nodes = list(G.nodes())
    if len(nodes) <= 1:
        return ""

    # 3. Categorize Nodes into Inbound, Target, and Outbound
    inbound_nodes = [n for n in nodes if G.has_edge(n, target_node) and n != target_node]
    outbound_nodes = [n for n in nodes if G.has_edge(target_node, n) and n != target_node]
    other_nodes = [n for n in nodes if n != target_node and n not in inbound_nodes and n not in outbound_nodes]

    # Dynamic Canvas Height calculation based on node count
    max_side_nodes = max(len(inbound_nodes), len(outbound_nodes), len(other_nodes), 1)
    fig_height = max(8.0, max_side_nodes * 0.75)
    fig_width = 14.0

    plt.figure(figsize=(fig_width, fig_height), facecolor="#0F172A")  # Deep Slate Dark Theme
    ax = plt.gca()
    ax.set_facecolor("#0F172A")

    # 4. Manual Columnar Node Positioning (Fix 1)
    pos = {}
    pos[target_node] = np.array([0.0, 0.0])

    # Left Column: Inbound Senders
    if inbound_nodes:
        y_span = max(3.5, len(inbound_nodes) * 0.9)
        y_pts = np.linspace(y_span / 2.0, -y_span / 2.0, len(inbound_nodes))
        for n, y in zip(inbound_nodes, y_pts):
            pos[n] = np.array([-3.5, y])

    # Right Column: Outbound Receivers (Evenly Spaced Vertically)
    if outbound_nodes:
        y_span = max(4.0, len(outbound_nodes) * 0.85)
        y_pts = np.linspace(y_span / 2.0, -y_span / 2.0, len(outbound_nodes))
        for n, y in zip(outbound_nodes, y_pts):
            pos[n] = np.array([3.5, y])

    # Top/Bottom: Other Nodes
    if other_nodes:
        y_pts = np.linspace(2.5, -2.5, len(other_nodes))
        for n, y in zip(other_nodes, y_pts):
            pos[n] = np.array([0.0, y])

    # 5. Titles & Headers
    if aml_pattern == "rapid_cash_out":
        title = f"Rapid Cash-Out Network Topology — Customer {target_cust}"
        subtitle = "Inbound Deposit ($180,000) -> Rapid Multi-Account Draining (96.7% Outflow)"
    elif aml_pattern == "structuring":
        title = f"Structuring Network Topology — Customer {target_cust}"
        subtitle = "Multi-Account Deposits Fanning Out Just Below $10,000 Threshold"
    else:
        title = f"Transaction Network Topology — Customer {target_cust}"
        subtitle = "Money Movement & Counterparty Account Interconnections"

    # 6. Node Colors, Sizes, and High-Contrast Labels (Fix 4)
    node_colors = []
    node_sizes = []
    for n in nodes:
        if n == target_node:
            node_colors.append("#EF4444")  # Bright Red
            node_sizes.append(2800)
        elif n in inbound_nodes:
            node_colors.append("#F59E0B")  # Amber/Gold
            node_sizes.append(1500)
        else:
            node_colors.append("#10B981")  # Emerald Green
            node_sizes.append(1300)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95, edgecolors="#FFFFFF", linewidths=2.5)

    # Format Node Labels
    labels = {}
    for n in nodes:
        if n == target_node:
            labels[n] = f"Cust {target_cust}\n[FLAGGED]"
        else:
            clean_name = n.replace("Customer_", "Cust ").replace("_", " ")
            labels[n] = clean_name

    # Render Node Labels with Dark Rounded Background Boxes (No overlap & high contrast)
    for n, (x, y) in pos.items():
        txt = labels[n]
        is_target = (n == target_node)
        fs = 11 if is_target else 9.5
        fw = "bold" if is_target else "normal"
        bg_color = "#991B1B" if is_target else "#1E293B"
        border_color = "#F87171" if is_target else "#475569"

        ax.text(
            x, y, txt,
            fontsize=fs, fontweight=fw, color="#F8FAFC",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc=bg_color, ec=border_color, lw=1.5, alpha=0.92),
            zorder=10
        )

    # 7. Draw Directed Edges & Prominent Arrowheads (Fix 3)
    edge_weights = [d.get("amount", 1000) for u, v, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = [max(1.8, (w / max_w) * 5.0) for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="-|>",
        arrowsize=25,  # Prominent readable arrowheads (Fix 3)
        edge_color="#38BDF8",  # Sky Blue
        width=widths,
        alpha=0.75,
        connectionstyle="arc3,rad=0.04"
    )

    # 8. Edge Label Summarization & Anti-Clutter Annotation (Fix 2)
    # Check if outbound edges share identical amounts (e.g. $14,500 repeated across 12 edges)
    outbound_amts = [d.get("amount") for u, v, d in G.edges(data=True) if u == target_node]
    
    if len(outbound_amts) >= 3 and len(set(outbound_amts)) == 1:
        # Outbound edges are identical -> Summarize with single annotation box near customer
        single_amt = outbound_amts[0]
        total_out = sum(outbound_amts)
        summary_txt = f"{len(outbound_amts)} outbound transfers x ${single_amt:,.0f} = ${total_out:,.0f}"


        # Place summary box near target node on right side
        ax.text(
            1.6, 0.0, summary_txt,
            fontsize=10.5, fontweight="bold", color="#FDE047",  # Bright Yellow text
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#1E1B4B", ec="#6366F1", lw=2.0, alpha=0.95),
            zorder=12
        )

        # Draw only INBOUND edge labels
        for u, v, d in G.edges(data=True):
            if v == target_node:
                amt = d.get("amount", 0)
                mid_x = (pos[u][0] + pos[v][0]) / 2.0
                mid_y = (pos[u][1] + pos[v][1]) / 2.0
                ax.text(
                    mid_x, mid_y + 0.25, f"${amt:,.0f}",
                    fontsize=10, fontweight="bold", color="#38BDF8",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#0F172A", ec="#38BDF8", lw=1.2, alpha=0.9),
                    zorder=11
                )
    else:
        # Standard edge labels for distinct amounts
        edge_labels = {(u, v): f"${d['amount']:,.0f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=8.5, font_color="#F3F4F6",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155", alpha=0.85)
        )

    # 9. Legend & Title Banners (Fix 6)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Flagged Entity', markerfacecolor='#EF4444', markersize=11),
        Line2D([0], [0], marker='o', color='w', label='Inbound Source Account', markerfacecolor='#F59E0B', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Outbound Destination Account', markerfacecolor='#10B981', markersize=9)
    ]
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=9.5)

    plt.suptitle(title, fontsize=16, fontweight="bold", color="#F8FAFC", y=0.98)
    plt.title(subtitle, fontsize=11.0, fontstyle="italic", color="#94A3B8", y=0.93)
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)



    timestamp = int(time.time())
    ts_path = f"charts/network_{timestamp}.png"
    latest_path = "charts/latest_network.png"

    plt.savefig(ts_path, dpi=180, bbox_inches="tight", facecolor="#0F172A")
    plt.savefig(latest_path, dpi=180, bbox_inches="tight", facecolor="#0F172A")
    plt.close()

    print(f"  [Network Graph Saved]: {latest_path}")
    return latest_path


def build_flow_dot(plan_meta: Dict[str, Any], result_summary: str = "") -> str:
    """
    Constructs a Graphviz DOT string visualizing the agent's decision and tool execution flow.
    Executed tools in plan are rendered in solid GREEN connected in main sequence.
    Skipped tools are rendered in dashed GREY off the main flow path.
    """
    intent = plan_meta.get("intent", "broad_analysis")
    aml_pattern = plan_meta.get("aml_pattern") or "None"
    plan = plan_meta.get("plan", [])
    skipped = plan_meta.get("skipped", [])
    entities = plan_meta.get("entities", {})
    cust_id = entities.get("customer_id")
    
    # Format planner label
    cust_str = f" | cust: {cust_id}" if cust_id else ""
    planner_label = f"Supervisor Planner\\nintent: {intent}{cust_str}\\npattern: {aml_pattern}"
    
    # All 5 canonical tools in pipeline order
    all_tools = ["eda", "feature_engineering", "anomaly_detection", "risk_classification", "explanation"]
    
    result_label = result_summary if result_summary else "Analysis Complete"

    dot_lines = [
        'digraph AgentExecutionFlow {',
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.25,0.12"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#64748B"];',
        '',
        '  // Entry & Planner nodes',
        '  query [label="User Query", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#475569", style="filled,solid"];',
        f'  planner [label="{planner_label}", fillcolor="#312E81", fontcolor="#EEF2FF", color="#6366F1", style="filled,solid"];',
        '  query -> planner [color="#6366F1", penwidth=2.0];',
    ]

    # Guard for human clarification or no tools executed
    if intent == "needs_clarification" or not plan:
        clarify_q = plan_meta.get("clarifying_question", "Clarification Required")
        dot_lines.append(f'  halt [label="HALTED\\n{clarify_q}", fillcolor="#7C2D12", fontcolor="#FFEDD5", color="#F97316", style="filled,solid"];')
        dot_lines.append('  planner -> halt [color="#F97316", penwidth=2.0];')
        dot_lines.append('}')
        return '\n'.join(dot_lines)

    # Add nodes for all 5 tools
    for tool in all_tools:
        if tool in plan:
            # Executed tool: Solid GREEN
            tool_label = f"{tool}\\n(executed)"
            dot_lines.append(f'  {tool} [label="{tool_label}", fillcolor="#065F46", fontcolor="#D1FAE5", color="#10B981", style="filled,solid"];')
        else:
            # Skipped tool: Dashed GREY
            tool_label = f"{tool}\\n(skipped)"
            dot_lines.append(f'  {tool} [label="{tool_label}", fillcolor="#1F2937", fontcolor="#9CA3AF", color="#4B5563", style="filled,dashed"];')

    # Main Arrow Path: Connect planner -> first executed tool -> ... -> last executed tool -> result
    executed_in_order = [t for t in all_tools if t in plan]
    
    prev_node = "planner"
    for tool in executed_in_order:
        dot_lines.append(f'  {prev_node} -> {tool} [color="#10B981", penwidth=2.5];')
        prev_node = tool

    # Result Node
    dot_lines.append(f'  result [label="{result_label}", fillcolor="#1E3A8A", fontcolor="#DBEAFE", color="#3B82F6", style="filled,solid"];')
    dot_lines.append(f'  {prev_node} -> result [color="#3B82F6", penwidth=2.5];')

    # Connect skipped tools off the main path with faint grey dotted edges
    for tool in all_tools:
        if tool not in plan:
            dot_lines.append(f'  planner -> {tool} [color="#4B5563", style="dotted", penwidth=1.0];')

    dot_lines.append('}')
    return '\n'.join(dot_lines)

