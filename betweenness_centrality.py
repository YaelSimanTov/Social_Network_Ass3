import pandas as pd
import networkx as nx
import networkit as nk
from operator import itemgetter

# טען נתונים
target = pd.read_csv('musae_facebook_target.csv')
edges = pd.read_csv('musae_facebook_edges.csv')

# צור גרף NetworkX
G = nx.Graph()
for node_id, page_type, page_name in zip(target['id'], target['page_type'], target['page_name']):
    G.add_node(node_id, page_type=page_type, page_name=page_name)
for u, v in zip(edges['id_1'], edges['id_2']):
    G.add_edge(u, v)

# המרת גרף ל-NetworKit (מקבילי)
nkG = nk.graph.Graph(n=len(G.nodes()), weighted=False, directed=False)
node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
idx_to_node = {idx: node for node, idx in node_to_idx.items()}
for u, v in G.edges():
    nkG.addEdge(node_to_idx[u], node_to_idx[v])

# חישוב betweenness centrality (בצורה מהירה ומקבילית)
print("Computing betweenness centrality...")
bc = nk.centrality.Betweenness(nkG, normalized=True)
bc.run()

# שליפת תוצאות
betweenness = {idx_to_node[i]: bc.score(i) for i in range(nkG.numberOfNodes())}
top_10 = sorted(betweenness.items(), key=itemgetter(1), reverse=True)[:10]

# הצגת 10 הצמתים המרכזיים
results = []
for node_id, centrality in top_10:
    page_type = G.nodes[node_id]['page_type']
    page_name = G.nodes[node_id]['page_name']
    results.append({
        'Node ID': node_id,
        'Page Name': page_name,
        'Page Type': page_type,
        'Betweenness Centrality': round(centrality, 5)
    })

results_df = pd.DataFrame(results)
print("\nTop 10 Nodes by Betweenness Centrality:\n")
print(results_df)
