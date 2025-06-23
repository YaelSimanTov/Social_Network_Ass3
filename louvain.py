import networkx as nx
import community as community_louvain
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
edges_df = pd.read_csv("musae_facebook_edges.csv")
targets_df = pd.read_csv("musae_facebook_target.csv")

# Build the graph
G = nx.Graph()
for node_id, page_type in zip(targets_df['id'], targets_df['page_type']):
    G.add_node(node_id, page_type=page_type)

for u, v in zip(edges_df['id_1'], edges_df['id_2']):
    G.add_edge(u, v)

# Optional: Add page_name if it exists
if 'page_name' in targets_df.columns:
    for node_id, name in zip(targets_df['id'], targets_df['page_name']):
        G.nodes[node_id]['page_name'] = name
else:
    print("Warning: 'page_name' column not found.")

# שלב 1: זיהוי קהילות באמצעות אלגוריתם Louvain
partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'community')

# שלב 2: יצירת טבלת נתונים עם הקהילה וסוג הדף של כל צומת
community_data = []
for node in G.nodes():
    community_data.append({
        'node': node,
        'community': G.nodes[node]['community'],
        'page_type': G.nodes[node].get('page_type', 'Unknown')  # הגנה למקרה שאין page_type
    })

community_df = pd.DataFrame(community_data)

# שלב 3: חישוב טבלת הצלבות בין סוגי דפים לקהילות 4 הגדולות ביותר
top_communities = community_df['community'].value_counts().head(4).index
community_summary = pd.crosstab(
    community_df[community_df['community'].isin(top_communities)]['page_type'],
    community_df[community_df['community'].isin(top_communities)]['community']
)

# שלב 4: הדמיה באמצעות heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(community_summary, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Page Types in Top 4 Communities")
plt.xlabel("Community ID")
plt.ylabel("Page Type")
plt.tight_layout()
plt.show()


# שלב נוסף: בדיקת מספר הקהילות שזוהו
num_communities = len(set(partition.values()))
print(f"Number of communities detected by Louvain algorithm: {num_communities}")


results = []
for node in G.nodes():
    comm = G.nodes[node]['community']
    external = sum(
        1 for neighbor in G.neighbors(node)
        if G.nodes[neighbor]['community'] != comm
    )
    results.append((node, external))

top_bridges = sorted(results, key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Nodes Bridging Communities:")
for node, external_links in top_bridges:
    print(f"Node {node}, External Connections: {external_links}, Type: {G.nodes[node].get('page_type')}, Name: {G.nodes[node].get('page_name', 'N/A')}")
