import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# --- Read input files ---
target = pd.read_csv('musae_facebook_target.csv')
edges = pd.read_csv('musae_facebook_edges.csv')

# --- Create an undirected graph ---
G = nx.Graph()

# Add nodes with page_type attribute
for node_id, category in zip(target['id'], target['page_type']):
    G.add_node(node_id, page_type=category)

# Add edges between nodes
for n1, n2 in zip(edges['id_1'], edges['id_2']):
    G.add_edge(n1, n2)

# =====================
# Question: Are pages more likely to connect with others of the same category (homophily)?
# =====================

# Step 1: Calculate observed homophily ratio (in parallel)

# Function to check if an edge connects nodes of the same page_type
def is_same_type(edge):
    u, v = edge
    return G.nodes[u]['page_type'] == G.nodes[v]['page_type']

# Use multithreading to check all edges in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(is_same_type, G.edges()))

# Count the number of same-type edges
same_type_edges = sum(results)

# Compute observed homophily ratio
homophily_ratio = same_type_edges / G.number_of_edges()

# Step 2: Calculate expected homophily ratio assuming randomness
type_counts = defaultdict(int)
for node in G.nodes():
    type_counts[G.nodes[node]['page_type']] += 1

# Count the number of same-type pairs
total_same_type_pairs = sum(count * (count - 1) / 2 for count in type_counts.values())
total_possible_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
expected_ratio = total_same_type_pairs / total_possible_pairs

# Print results
print(f"Observed Homophily: {homophily_ratio:.3f}")
print(f"Expected Homophily (Random): {expected_ratio:.3f}")

# Step 3: Visualize cross-category connections using a heatmap
cross_matrix = pd.crosstab(
    [G.nodes[u]['page_type'] for u, v in G.edges()],
    [G.nodes[v]['page_type'] for u, v in G.edges()]
)

plt.figure(figsize=(10, 8))
sns.heatmap(cross_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Cross-Category Connections")
plt.xlabel("Page Type (Destination)")
plt.ylabel("Page Type (Source)")
plt.tight_layout()
plt.show()
