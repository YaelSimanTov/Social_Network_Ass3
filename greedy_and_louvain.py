import networkx as nx
import networkx.algorithms.community as nx_comm
import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from make_my_graph_weighted import largest_connected_component_subgraph

# קריאת קובצי הקלט
target = pd.read_csv('musae_facebook_target.csv')
edges  = pd.read_csv('musae_facebook_edges.csv')

# יצירת גרף חדש
G = nx.Graph()

# הוספת הצמתים לגרף עם תכונת page_type
for it, cat in zip(target['id'], target['page_type']):
    G.add_node(it, page_type=cat)

# הוספת הקשתות לגרף
for n1, n2 in zip(edges['id_1'], edges['id_2']):
    G.add_edge(n1, n2)

G = largest_connected_component_subgraph(G)


# --- Louvain ו-Greedy Community Detection ---
list_community_sets_greedy = list(nx_comm.greedy_modularity_communities(G))
print(list_community_sets_greedy[0:20])

# ממפה צומת → מספר קהילה
partition_greedy = {}
for i, comm in enumerate(list_community_sets_greedy):
    print("Community:", i)
    print("Number of elements:", len(comm))
    for n in comm:
        partition_greedy[n] = i

print(list(partition_greedy.items())[0:20])

# ציור גרף עם צבע לפי קהילה (greedy)
pos = nx.spring_layout(G)
cmap = cm.get_cmap('tab20', max(partition_greedy.values()) + 1)
plt.figure(figsize=(15, 15))
nx.draw_networkx_nodes(G, pos, node_size=20, cmap=cmap,
                       node_color=[partition_greedy[n] for n in G.nodes()])
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Greedy Modularity Communities")
plt.show()

# Louvain
partition_library = community_louvain.best_partition(G)
print(list(partition_library.items())[0:20])

# --- ניתוח ממוצע דרגה לפי קטגוריה ---
degree_df = pd.DataFrame({
    'node': list(G.nodes()),
    'degree': [G.degree(n) for n in G.nodes()]
})
degree_df['page_type'] = degree_df['node'].map(nx.get_node_attributes(G, 'page_type'))

# תרשים עמודות של דרגה ממוצעת לפי קטגוריה
plt.figure(figsize=(10, 5))
sns.barplot(data=degree_df, x='page_type', y='degree', estimator=np.mean)
plt.title("Average Degree by Page Type")
plt.xlabel("Page Type")
plt.ylabel("Average Degree")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- מבחן שונות (ANOVA) ---
categories = degree_df.groupby('page_type')['degree'].apply(list)
f_stat, p_value = f_oneway(*categories)
print(f"ANOVA p-value: {p_value:.4f} (Significant if < 0.05)")

# ציור גרף לפי קהילות Louvain
cmap = cm.get_cmap('tab20', max(partition_library.values()) + 1)
plt.figure(figsize=(15, 15))
nx.draw_networkx_nodes(G, pos, node_size=20, cmap=cmap,
                       node_color=[partition_library[n] for n in G.nodes()])
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Louvain Communities")
plt.show()

# --- הפיכת partition של Louvain לרשימת קהילות ---
num_comms = max(partition_library.values()) + 1
list_community_sets_library = [set() for _ in range(num_comms)]
for node, comm_id in partition_library.items():
    list_community_sets_library[comm_id].add(node)

# --- מדדי איכות לקהילות ---
for method_name, community_list in zip(['Greedy', 'Louvain'], [list_community_sets_greedy, list_community_sets_library]):
    print(f"{method_name} Communities:")
    print("  Coverage:", nx_comm.coverage(G, community_list))
    print("  Modularity:", nx_comm.modularity(G, community_list, weight='weight'))
    print("  Performance:", nx_comm.performance(G, community_list))
    print("---")

# --- תרשים מספר צמתים בכל קהילה בלוביין ---
pairs = []
for i, comm_nodes in enumerate(list_community_sets_library):
    print(f"Community {i}: {len(comm_nodes)} nodes")
    pairs.append((i, len(comm_nodes)))

community_index = [str(c) for c, _ in pairs]
number_of_nodes = [n for _, n in pairs]

plt.figure(figsize=(10, 8))
plt.bar(community_index, number_of_nodes)
plt.xlabel("Community")
plt.ylabel("Number of Nodes")
plt.title("Community Sizes (Louvain)")
plt.show()
