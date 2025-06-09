import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from make_my_graph_weighted import largest_connected_component_subgraph


# 1. קריאת הקבצים
target = pd.read_csv('musae_facebook_target.csv')
edges = pd.read_csv('musae_facebook_edges.csv')

# 2. יצירת הגרף
G = nx.Graph()

# הוספת צמתים עם סוג הדף כמאפיין
for node_id, page_type in zip(target['id'], target['page_type']):
    G.add_node(node_id, page_type=page_type)

# הוספת קשתות (גרף לא מכוון)
for n1, n2 in zip(edges['id_1'], edges['id_2']):
    G.add_edge(n1, n2)

G = largest_connected_component_subgraph(G)
# בדיקת מספר צמתים וקשתות
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# 3. יצירת מטריצת סמיכוּת (adjacency matrix) בגודל מספר הצמתים
n = G.number_of_nodes()
adj_matrix = np.zeros((n, n), dtype=int)

for n1, n2 in zip(edges['id_1'], edges['id_2']):
    adj_matrix[n1][n2] = 1
    adj_matrix[n2][n1] = 1  # הגרף הוא לא מכוון

# 4. הרצת KMeans על מטריצת הסמיכות
km = KMeans(n_clusters=4, random_state=0)
km.fit(adj_matrix)

# 5. תצוגת פלט של תוויות האשכולות
print("Shape of labels:", km.labels_.shape)

# 6. ויזואליזציה – היסטוגרמה של התפלגות הצמתים בין הקבוצות
plt.figure(figsize=(8, 4))
plt.hist(km.labels_, bins=np.arange(5)-0.5, rwidth=0.8)
plt.title("Cluster Distribution (KMeans)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Nodes")
plt.xticks([0, 1, 2, 3])
plt.grid(True)
plt.show()
