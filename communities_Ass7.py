import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from make_my_graph_weighted import largest_connected_component_subgraph

import community as community_louvain
from networkx.algorithms import community as nx_comm
from networkx.algorithms.community import girvan_newman
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def draw_communities(G, communities, title="Community Detection", layout='spring'):
    """
    Draws a graph with nodes colored by community.

    Parameters:
    - G: NetworkX graph
    - communities: Either a dictionary (node -> community_id) or a list of sets (each set is a group of nodes)
    - title: Title for the plot
    - layout: Graph layout algorithm - options: 'spring', 'kamada', 'spectral', 'circular'
    """
    if isinstance(communities, dict):
        color_map = communities
    else:
        color_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                color_map[node] = i

    if layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    colors = [color_map[node] for node in G.nodes()]

    plt.figure(figsize=(12, 10))
    nx.draw(
        G, pos, node_color=colors, with_labels=False,
        node_size=200, cmap = plt.colormaps['tab20']
        , alpha=1, edgecolors='black', linewidths=0.8
    )
    plt.title(title, fontsize=16)
    plt.show()

def compare_communities(true_labels, detected_labels):
    """
    Compare true community labels with detected community labels.

    Parameters:
    - true_labels: list of true labels (ground truth) for each node
    - detected_labels: list of labels detected by community detection algorithm

    Returns:
    - dict with ARI and NMI scores
    """
    ari = adjusted_rand_score(true_labels, detected_labels)
    nmi = normalized_mutual_info_score(true_labels, detected_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    return {'ARI': ari, 'NMI': nmi}




def compare_communities(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi


def louvain_communities_labels(G):
    partition = community_louvain.best_partition(G)
    # מחזיר רשימת תוויות לפי סדר צמתים ב-G.nodes()
    return [partition[node] for node in G.nodes()]


def greedy_communities_labels(G):
    communities = list(nx_comm.greedy_modularity_communities(G))
    # ממיר רשימת קהילות לרשימת תוויות לפי סדר הצמתים
    label_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            label_map[node] = i
    return [label_map[node] for node in G.nodes()]


def girvan_newman_communities_labels(G, level=1):
    comp = girvan_newman(G)
    for i in range(level):
        communities = next(comp)
    label_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            label_map[node] = i
    return [label_map[node] for node in G.nodes()]


def evaluate_algorithms(G, true_labels):
    algorithms = {
        "Louvain": louvain_communities_labels,
        "Greedy Modularity": greedy_communities_labels,
        "Girvan-Newman (level 1)": lambda g: girvan_newman_communities_labels(g, level=1)
    }

    for name, func in algorithms.items():
        predicted_labels = func(G)
        ari, nmi = compare_communities(true_labels, predicted_labels)
        print(f"{name}:\n  Adjusted Rand Index (ARI): {ari:.4f}\n  Normalized Mutual Information (NMI): {nmi:.4f}\n")

        communities_dict = {node: label for node, label in zip(G.nodes(), predicted_labels)}
        # ציור הגרף עם הקהילות
        draw_communities(G, communities_dict, title=f"Communities detected by {name}")


"""**********************************************************************
**********************************************************************
Network of Thrones
**********************************************************************
*********************************************************************"""

# edges_df = pd.read_csv('stormofswords.csv')
# tribes_df = pd.read_csv('tribes.csv', header=None, names=['node', 'tribe'])
#
# # יצירת הגרף
# G = nx.Graph()
# for idx, row in edges_df.iterrows():
#     G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
#
# true_label_dict = dict(zip(tribes_df['node'], tribes_df['tribe']))
# true_labels = [true_label_dict.get(node, -1) for node in G.nodes()]

# evaluate_algorithms(G, true_labels)
"""**********************************************************************
   **********************************************************************
   My Facebook Network - Subgraph of only Politician and TV-show pages
   **********************************************************************
*********************************************************************"""

# # קריאת קובצי הקלט
target = pd.read_csv('musae_facebook_target.csv')
edges = pd.read_csv('musae_facebook_edges.csv')

# יצירת גרף חדש
G = nx.Graph()

# הוספת הצמתים לגרף עם תכונת page_type
for it, cat in zip(target['id'], target['page_type']):
    G.add_node(it, page_type=cat)

# הוספת הקשתות לגרף
for n1, n2 in zip(edges['id_1'], edges['id_2']):
    G.add_edge(n1, n2)

G = largest_connected_component_subgraph(G)
# true_labels = dict(zip(target['id_facebook'], target['type_page']))
# יצירת true_labels לפי page_type
true_label_dict = dict(zip(target['id'], target['page_type']))
true_labels = [true_label_dict.get(node, -1) for node in G.nodes()]

evaluate_algorithms(G, true_labels)



