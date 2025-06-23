import pandas as pd
import networkx as nx
from make_my_graph_weighted import largest_connected_component_subgraph
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from scipy.stats import linregress
from count_edges import remove_self_loops,count_self_loops
def plot_log_degree_distribution(G):
    """
    Plots the degree distribution of graph G on a log-log scale.
    X-axis: Degree (log)
    Y-axis: Probability Density (log)
    """

    # Calculate degrees
    degrees = [G.degree(n) for n in G.nodes()]

    # Calculate degree frequency distribution
    degree_counts = pd.Series(degrees).value_counts().sort_index()

    # Normalize to get probability density
    x = degree_counts.index
    y = degree_counts.values / degree_counts.values.sum()  # PDF

    # === Print the top 5 highest-degree nodes ===
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes with the highest degree:")
    for node, degree in top_nodes:
        name = G.nodes[node].get('page_name', 'Unknown')
        print(f"Node ID: {node}, Degree: {degree}, Name: {name}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(x, y, width=0.8, color='skyblue', edgecolor='black')

    # Set both axes to log scale
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Degree (log scale)")
    plt.ylabel("Probability Density (log scale)")
    plt.title("Degree Distribution (Log-Log Scale)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()






# =====================
# Degree Distribution Histogram
# =====================

# Load the data
edges_df = pd.read_csv("musae_facebook_edges.csv")
targets_df = pd.read_csv("musae_facebook_target.csv")


# Add 'page_name' to the graph (if available)

# Build the graph
G = nx.Graph()
for node_id, page_type in zip(targets_df['id'], targets_df['page_type']):
    G.add_node(node_id, page_type=page_type)

for u, v in zip(edges_df['id_1'], edges_df['id_2']):
    G.add_edge(u, v)


if 'page_name' in targets_df.columns:
    for node_id, name in zip(targets_df['id'], targets_df['page_name']):
        G.nodes[node_id]['page_name'] = name
else:
    print("Warning: 'page_name' column not found in target CSV.")

# plot_log_degree_distribution(G)

# Calculate degrees and sort descending
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]

# Print top 10 nodes with degree, page_type, and page_name
print("Top 10 nodes with highest degree:\n")
for node, degree in top_nodes:
    page_type = G.nodes[node].get('page_type', 'Unknown')
    page_name = G.nodes[node].get('page_name', 'Unknown')
    print(f"Node ID: {node}, Degree: {degree}, Type: {page_type}, Name: {page_name}")

# Count nodes with self-loops
print("Nodes with self-loops:", count_self_loops(G))  # Output: 2

# Remove self-loops
removed = remove_self_loops(G)
print("Removed self-loops:", removed)  # Output: 2


"""
********************************************************************************************
Create subgraphs by category
********************************************************************************************
"""

# ************************* Identify nodes per category  *************************
politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()
government_nodes = targets_df[targets_df['page_type'] == 'government']['id'].tolist()
tvshow_nodes = targets_df[targets_df['page_type'] == 'tvshow']['id'].tolist()
company_nodes = targets_df[targets_df['page_type'] == 'company']['id'].tolist()


# ************************* Create subgraphs by category  *************************
G_politician = G.subgraph(politician_nodes)
G_government = G.subgraph(government_nodes)
G_tvshow = G.subgraph(tvshow_nodes)
G_company = G.subgraph(company_nodes)

# Layout formerly each subgraph (fixed for consistency)
pos_politician = nx.spring_layout(G_politician, seed=42)
pos_government = nx.spring_layout(G_government, seed=42)
pos_tvshow = nx.spring_layout(G_tvshow, seed=42)
pos_company = nx.spring_layout(G_company, seed=42)

plot_log_degree_distribution(G_politician)
plot_log_degree_distribution(G_government)
plot_log_degree_distribution(G_tvshow)
plot_log_degree_distribution(G_company)
#


# ************************* Draw each subgraph separately *************************
# Politician
plt.figure(figsize=(12, 8))
nx.draw(G_politician, pos=pos_politician, with_labels=False, node_size=30, node_color='red')
plt.title("Political Pages Subgraph (Red for Politicians)")
plt.show()

# # Government
# plt.figure(figsize=(12, 8))
# nx.draw(G_government, pos=pos_government, with_labels=False, node_size=30, node_color='blue')
# plt.title("Government Pages Subgraph (Blue for Government)")
# plt.show()
#
# # TV Show
# plt.figure(figsize=(12, 8))
# nx.draw(G_tvshow, pos=pos_tvshow, with_labels=False, node_size=30, node_color='yellow')
# plt.title("TV Show Pages Subgraph (Yellow for TV Shows)")
# plt.show()
#
# # Company
# plt.figure(figsize=(12, 8))
# nx.draw(G_company, pos=pos_company, with_labels=False, node_size=30, node_color='green')
# plt.title("Company Pages Subgraph (Green for Companies)")
# plt.show()
#
#
