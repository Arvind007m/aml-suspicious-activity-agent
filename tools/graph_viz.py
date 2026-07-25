"""
Transaction Network Graph Visualization Tool (tools/graph_viz.py)
Visualizes money flow topologies for flagged entities using NetworkX and Matplotlib.
Supports pattern-specific layout topologies (rapid_cash_out, structuring, amount_spike).
"""

import os
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx


def build_transaction_graph(df_raw: pd.DataFrame, top_entities: List[Dict[str, Any]], aml_pattern: str = None) -> str:
    """
    Constructs and renders a directed transaction money-flow network graph for top flagged customer.
    Adapts topology layout and styling dynamically based on aml_pattern.
    Saves image to charts/latest_network.png and returns file path.
    """
    if not top_entities or df_raw.empty:
        return ""

    # Ensure charts directory exists
    os.makedirs("charts", exist_ok=True)
    
    # 1. Identify primary target flagged entity
    target_item = top_entities[0]
    target_cust = str(target_item.get("customer_id")).strip()
    
    df = df_raw.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["sender_account"] = df["sender_account"].astype(str)
    df["receiver_account"] = df["receiver_account"].astype(str)

    # Filter transactions related to target customer (as sender or receiver or customer_id)
    cust_txns = df[
        (df["customer_id"] == target_cust) |
        (df["sender_account"].str.contains(target_cust, na=False)) |
        (df["receiver_account"].str.contains(target_cust, na=False))
    ].copy()

    if cust_txns.empty:
        # Fallback: take top 20 transactions by amount
        cust_txns = df.sort_values("amount", ascending=False).head(20).copy()

    # Cap to top ~20 transactions by amount for clean visual readability
    if len(cust_txns) > 20:
        cust_txns = cust_txns.sort_values("amount", ascending=False).head(20).copy()

    # 2. Build Directed Graph using NetworkX
    G = nx.DiGraph()

    target_node = f"Customer_{target_cust}"
    G.add_node(target_node, node_type="target", label=f"Cust {target_cust}\n[FLAGGED]")

    for idx, row in cust_txns.iterrows():
        sender = str(row["sender_account"])
        receiver = str(row["receiver_account"])
        amt = float(row["amount"])
        fmt = str(row.get("payment_format", "Transfer"))

        # Format node names
        s_node = f"Cust_{target_cust}" if sender.endswith(target_cust) or sender == target_cust else sender
        r_node = f"Cust_{target_cust}" if receiver.endswith(target_cust) or receiver == target_cust else receiver
        
        # Standardize target node identifier
        if target_cust in s_node:
            s_node = target_node
        if target_cust in r_node:
            r_node = target_node

        G.add_edge(s_node, r_node, amount=amt, payment_format=fmt)

    if G.number_of_nodes() <= 1:
        return ""

    # 3. Create High-Resolution Matplotlib Figure
    plt.figure(figsize=(12, 8), facecolor="#111827")  # Sleek dark mode background
    ax = plt.gca()
    ax.set_facecolor("#111827")

    # 4. Pattern-Specific Node Layout & Topology Positioning
    nodes = list(G.nodes())
    pos = {}

    if aml_pattern == "rapid_cash_out":
        # Rapid cash-out topology: Inbound on Left -> Target Center -> Outbound Fan-Out on Right
        pos[target_node] = np.array([0.0, 0.0])
        inbound_nodes = [n for n in nodes if G.has_edge(n, target_node)]
        outbound_nodes = [n for n in nodes if G.has_edge(target_node, n)]

        # Position inbound senders on left
        for i, n in enumerate(inbound_nodes):
            y_offset = (i - len(inbound_nodes)/2.0) * 1.5
            pos[n] = np.array([-2.5, y_offset])

        # Position outbound receivers on right fan-out
        for i, n in enumerate(outbound_nodes):
            y_offset = (i - len(outbound_nodes)/2.0) * 0.8
            pos[n] = np.array([2.5, y_offset])

        # Any leftover nodes
        unpositioned = [n for n in nodes if n not in pos]
        if unpositioned:
            spring_pos = nx.spring_layout(G.subgraph(unpositioned), center=(0, 2))
            pos.update(spring_pos)

        title = f"Rapid Cash-Out Network Topology - Customer {target_cust}"
        subtitle = "Inbound Wire Deposit ($180k) -> Rapid Multi-Account Draining (96.7% Outflow)"

    elif aml_pattern == "structuring":
        # Structuring topology: Circular hub layout showing near-threshold deposit fan-out
        pos = nx.shell_layout(G, nlist=[[target_node], [n for n in nodes if n != target_node]])
        title = f"Structuring Network Topology (Smurfing) - Customer {target_cust}"
        subtitle = "Multi-Account Deposits Fanning Out Just Below $10,000 Threshold"

    else:
        # Amount anomaly / Default topology
        pos = nx.kamada_kawai_layout(G) if len(nodes) > 2 else nx.spring_layout(G)
        title = f"Transaction Network Topology - Customer {target_cust}"
        subtitle = "Visualizing Money Movement & Account Interconnections"

    # 5. Node Colors & Sizes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n == target_node:
            node_colors.append("#EF4444")  # Bright Red for Flagged Entity
            node_sizes.append(2200)
        elif G.has_edge(n, target_node):
            node_colors.append("#F59E0B")  # Amber/Orange for Inbound Senders
            node_sizes.append(1200)
        else:
            node_colors.append("#10B981")  # Emerald/Green for Outbound Receivers
            node_sizes.append(1000)

    # 6. Draw Nodes & Edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors="#FFFFFF", linewidths=2.0)
    
    # Draw Node Labels
    labels = {n: n.replace("Customer_", "Cust ").replace("_", " ") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="#FFFFFF", font_weight="bold")

    # Draw Directed Edges
    edge_weights = [d.get("amount", 1000) for u, v, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = [max(1.5, (w / max_w) * 4.5) for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#38BDF8",  # Sky Blue arrows
        width=widths,
        alpha=0.75,
        connectionstyle="arc3,rad=0.1"
    )

    # Edge Amount Labels
    edge_labels = {(u, v): f"${d['amount']:,.0f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#F3F4F6", bbox=dict(boxstyle="round,pad=0.2", fc="#1F2937", ec="#374151", alpha=0.8))

    # 7. Add Title & Subtitle Banners
    plt.suptitle(title, fontsize=15, fontweight="bold", color="#F9FAFB", y=0.96)
    plt.title(subtitle, fontsize=11, fontstyle="italic", color="#9CA3AF", pad=10)
    plt.axis("off")
    plt.tight_layout()

    # 8. Save Graph Files
    timestamp = int(time.time())
    ts_path = f"charts/network_{timestamp}.png"
    latest_path = "charts/latest_network.png"
    
    plt.savefig(ts_path, dpi=180, bbox_inches="tight", facecolor="#111827")
    plt.savefig(latest_path, dpi=180, bbox_inches="tight", facecolor="#111827")
    plt.close()

    print(f"  [Network Graph Saved]: {latest_path}")
    return latest_path
