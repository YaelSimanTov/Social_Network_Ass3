from sklearn.cluster import SpectralClustering
import numpy as np
import networkx as nx
import pandas as pd


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


# Get adjacency matrix
A = nx.to_numpy_array(G)

# Apply spectral clustering with k=4
sc = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
labels = sc.fit_predict(A)

# Assign cluster label to each node
for node, label in zip(G.nodes(), labels):
    G.nodes[node]['community'] = label


import matplotlib.pyplot as plt

# Define colors for communities
color_map = ['red', 'blue', 'green', 'orange']
node_colors = [color_map[G.nodes[node]['community']] for node in G.nodes()]

# Draw the network
plt.figure(figsize=(12, 8))
nx.draw_spring(G, node_color=node_colors, node_size=50, with_labels=False)
plt.title("Graph divided into 4 communities (Spectral Clustering)")
plt.show()


import seaborn as sns

# Create DataFrame with node attributes: community and page_type
data = []
for node in G.nodes():
    data.append({
        'node': node,
        'community': G.nodes[node]['community'],
        'page_type': G.nodes[node].get('page_type', 'Unknown')
    })

df = pd.DataFrame(data)

# Crosstab of page_type vs community
crosstab = pd.crosstab(df['page_type'], df['community'])

# # Visualization of crosstab
# plt.figure(figsize=(10, 6))
# sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt='d')
# plt.title("Distribution of Page Types Across 4 Communities")
# plt.xlabel("Community")
# plt.ylabel("Page Type")
# plt.tight_layout()
# plt.show()


# # Calculate for each node how many neighbors belong to different communities
# results = []
# for node in G.nodes():
#     node_comm = G.nodes[node]['community']
#     external_neighbors = sum(
#         1 for neighbor in G.neighbors(node)
#         if G.nodes[neighbor]['community'] != node_comm
#     )
#     results.append((node, external_neighbors))
#
# # Show top 10 nodes that bridge communities
# top_bridges = sorted(results, key=lambda x: x[1], reverse=True)[:10]
# print("Top 10 nodes bridging communities:")
# for node, count in top_bridges:
#     print(f"Node {node}, connections to other communities: {count}, Page type: {G.nodes[node].get('page_type')}")


# # Identify top bridging nodes between communities
# results = []
# for node in G.nodes():
#     comm = G.nodes[node]['community']
#     external = sum(
#         1 for neighbor in G.neighbors(node)
#         if G.nodes[neighbor]['community'] != comm
#     )
#     results.append((node, external))
#
# top_bridges = sorted(results, key=lambda x: x[1], reverse=True)[:10]
# print("\nTop 10 Nodes Bridging Communities:")
# for node, external_links in top_bridges:
#     print(f"Node {node}, External Connections: {external_links}, Type: {G.nodes[node].get('page_type')}, Name: {G.nodes[node].get('page_name', 'N/A')}")

# Count how many neighbors each node has from *other* communities
inter_community_links = []

for node in G.nodes():
    node_comm = G.nodes[node]['community']
    external_links = sum(
        1 for neighbor in G.neighbors(node)
        if G.nodes[neighbor]['community'] != node_comm
    )
    inter_community_links.append((node, external_links))

# Sort by number of external links
top_bridging_nodes = sorted(inter_community_links, key=lambda x: x[1], reverse=True)[:10]

# Display the results
print("Top 10 Nodes Bridging Communities:")
for node_id, external in top_bridging_nodes:
    page_type = G.nodes[node_id].get('page_type', 'N/A')
    name = G.nodes[node_id].get('page_name', 'N/A')
    print(f"Node {node_id}, External Connections: {external}, Type: {page_type}, Name: {name}")
