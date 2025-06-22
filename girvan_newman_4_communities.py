import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import girvan_newman

# Step 1: Load data
print("Loading data...")
target = pd.read_csv('musae_facebook_target.csv')
edges = pd.read_csv('musae_facebook_edges.csv')

# Step 2: Build the graph
print("Building graph...")
G = nx.Graph()
for node_id, cat in zip(target['id'], target['page_type']):
    G.add_node(node_id, page_type=cat)
for u, v in zip(edges['id_1'], edges['id_2']):
    G.add_edge(u, v)

# Step 3: Detect communities using Girvan–Newman until 4 communities are reached
print("Detecting 4 communities...")
comp_gen = girvan_newman(G)

# We loop until the number of communities == 4
for communities in comp_gen:
    if len(communities) == 4:
        break

# Step 4: Assign community ID to each node
print("Assigning community labels...")
node_to_community = {}
for i, community_nodes in enumerate(communities):
    for node in community_nodes:
        node_to_community[node] = i

nx.set_node_attributes(G, node_to_community, 'community')

# Step 5: Build dataframe of node attributes
print("Creating dataframe of nodes and attributes...")
data = []
for node in node_to_community:  # Use only assigned nodes
    data.append({
        'node': node,
        'community': node_to_community[node],
        'page_type': G.nodes[node].get('page_type', 'Unknown')
    })

df = pd.DataFrame(data)

# Step 6: Crosstab of community vs page_type
print("Creating crosstab...")
community_summary = pd.crosstab(df['page_type'], df['community'])

# Step 7: Visualization
print("Visualizing result...")
plt.figure(figsize=(10, 6))
sns.heatmap(community_summary, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Distribution of Page Types Across 4 Communities")
plt.xlabel("Community ID")
plt.ylabel("Page Type")
plt.tight_layout()
plt.show()

print("Done.")
