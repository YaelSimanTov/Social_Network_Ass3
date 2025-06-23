import pandas as pd
import networkx as nx
from make_my_graph_weighted import largest_connected_component_subgraph
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from scipy.stats import linregress


def count_edges_between_categories(graph, targets_df, category1, category2):
    """
    Count how many edges in the graph connect nodes from category1 to nodes from category2.

    Parameters:
        graph (networkx.Graph): The full undirected graph
        targets_df (pd.DataFrame): DataFrame containing 'id' and 'page_type' columns
        category1 (str): The first category (e.g. 'politician')
        category2 (str): The second category (e.g. 'government')

    Returns:
        int: Number of edges between the two categories
    """
    # Create sets of node IDs for each category
    nodes_cat1 = set(targets_df[targets_df['page_type'] == category1]['id'])
    nodes_cat2 = set(targets_df[targets_df['page_type'] == category2]['id'])

    count = 0
    for u, v in graph.edges():
        if (u in nodes_cat1 and v in nodes_cat2) or (v in nodes_cat1 and u in nodes_cat2):
            count += 1
    return count



def count_edges_in_graph(G):
    """
    This function receives a NetworkX graph and returns the number of edges in it.
    """
    return G.number_of_edges()

def count_self_loops(G):
    """
    Counts how many nodes in the graph have self-loop edges.
    """
    self_loops = list(nx.selfloop_edges(G))
    nodes_with_self_loops = set([u for u, v in self_loops])
    return len(nodes_with_self_loops)

def remove_self_loops(G):
    """
    Removes all self-loop edges (edges from a node to itself) from the graph.
    """
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    return len(self_loops)


# =====================
#Amount of edges connect politicians to government organizations
# =====================

# Load the data
edges_df = pd.read_csv("musae_facebook_edges.csv")
targets_df = pd.read_csv("musae_facebook_target.csv")

# Build the full undirected graph (with all nodes)
G_full = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')


print(f"Number of edges of the full graph: {count_edges_in_graph(G_full)}")
G = largest_connected_component_subgraph(G_full)


# Count nodes with self-loops
print("Nodes with self-loops:", count_self_loops(G))  # Output: 2

# Remove self-loops
removed = remove_self_loops(G)
print("Removed self-loops:", removed)  # Output: 2

# Verify again
print("Nodes with self-loops after removal:", count_self_loops(G))  # Output: 0
print(f"Number of edges of the largest connected component subgraph: {count_edges_in_graph(G)}")



# List of categories that appear in the dataset
categories = targets_df['page_type'].unique().tolist()

# Dictionary to store counts
dic_count_edges_between_categories = {}

# Loop over all combinations (with repetition) of categories
for cat1, cat2 in itertools.combinations_with_replacement(categories, 2):
    count = count_edges_between_categories(G, targets_df, cat1, cat2)
    dic_count_edges_between_categories[(cat1, cat2)] = count

# Print the dictionary
print(dic_count_edges_between_categories)
