import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
#
# def edge_removal_giant_component(G, strategy="weak_to_strong"):
#     G_copy = G.copy()
#
#     if strategy == "weak_to_strong":
#         # Sort edges from weak to strong (ascending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'])
#     elif strategy == "strong_to_weak":
#         # Sort edges from strong to weak (descending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
#     elif strategy == "random":
#         # Shuffle edges randomly
#         edges_sorted = list(G_copy.edges(data=True))
#         random.shuffle(edges_sorted)
#     elif strategy == "betweenness":
#         # Sort edges based on betweenness centrality
#         betweenness = nx.edge_betweenness_centrality(G_copy)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: betweenness[(x[0], x[1])], reverse=True)
#     else:
#         raise ValueError(
#             "Unknown strategy. Choose from 'weak_to_strong', 'strong_to_weak', 'random', or 'betweenness'.")
#
#     giant_sizes = []
#     removed_edges = []  # Track the number of removed edges
#     num_edges = len(edges_sorted)
#
#     for i, (u, v, d) in enumerate(edges_sorted):
#         G_copy.remove_edge(u, v)
#         removed_edges.append(i + 1)  # Count the number of removed edges
#         if len(G_copy) == 0 or G_copy.number_of_edges() == 0:
#             giant_sizes.append(0)
#         else:
#             giant_sizes.append(len(max(nx.connected_components(G_copy), key=len)))
#
#     return removed_edges, giant_sizes
#
#
#
#
# # --- Creating a subplot with 4 different strategies ---
# def plot_edge_removal_strategies(G):
#     strategies = ['weak_to_strong', 'strong_to_weak', 'random', 'betweenness']
#     colors = ['b', 'g', 'r', 'c']  # Different colors for each strategy
#     labels = ['Weak to Strong', 'Strong to Weak', 'Random', 'Betweenness']
#
#     plt.figure(figsize=(12, 10))
#
#     for i, strategy in enumerate(strategies):
#         x, y = edge_removal_giant_component(G, strategy=strategy)
#         plt.subplot(2, 2, i + 1)  # Create 2x2 grid of subplots
#         plt.plot(x, y, marker='o', color=colors[i], label=labels[i])
#         plt.xlabel('Number of edges removed')
#         plt.ylabel('Size of the giant component')
#         plt.title(f'{labels[i]} - Network of Thrones')
#         plt.grid()
#         plt.legend()
#
#     plt.tight_layout()  # Adjust the layout to prevent overlap
#     plt.show()



################################################################
# --- General function to remove edges based on selected strategy ---
# def edge_removal_giant_component(G, strategy="weak_to_strong"):
#     G_copy = G.copy()
#
#     if strategy == "weak_to_strong":
#         # Sort edges from weak to strong (ascending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'])
#     elif strategy == "strong_to_weak":
#         # Sort edges from strong to weak (descending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
#     elif strategy == "random":
#         # Shuffle edges randomly
#         edges_sorted = list(G_copy.edges(data=True))
#         random.shuffle(edges_sorted)
#     elif strategy == "betweenness":
#         # Sort edges based on betweenness centrality
#         betweenness = nx.edge_betweenness_centrality(G_copy)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: betweenness[(x[0], x[1])], reverse=True)
#     else:
#         raise ValueError(
#             "Unknown strategy. Choose from 'weak_to_strong', 'strong_to_weak', 'random', or 'betweenness'.")
#
#     giant_sizes = []
#     percents_removed = []
#     num_edges = len(edges_sorted)
#
#     for i, (u, v, d) in enumerate(edges_sorted):
#         G_copy.remove_edge(u, v)
#         if len(G_copy) == 0 or G_copy.number_of_edges() == 0:
#             giant_sizes.append(0)
#         else:
#             giant_sizes.append(len(max(nx.connected_components(G_copy), key=len)))
#             ####################################################
#         percents_removed.append((i + 1) / num_edges)
#
#     return percents_removed, giant_sizes
#
#
# # --- Creating a subplot with 4 different strategies ---
# def plot_edge_removal_strategies(G):
#     strategies = ['weak_to_strong', 'strong_to_weak', 'random', 'betweenness']
#     colors = ['b', 'g', 'r', 'c']  # Different colors for each strategy
#     labels = ['Weak to Strong', 'Strong to Weak', 'Random', 'Betweenness']
#
#     plt.figure(figsize=(12, 10))
#
#     for i, strategy in enumerate(strategies):
#         x, y = edge_removal_giant_component(G, strategy=strategy)
#         plt.subplot(2, 2, i + 1)  # Create 2x2 grid of subplots
#         plt.plot(x, y, marker='o', color=colors[i], label=labels[i])
#         plt.xlabel('Percentage of edges removed')
#         plt.ylabel('Size of the giant component')
#         plt.title(f'{labels[i]} - Network of Thrones')
#         plt.grid()
#         plt.legend()
#
#     plt.tight_layout()  # Adjust the layout to prevent overlap
#     plt.show()
#

########################################################################

# # --- General function to remove edges based on selected strategy ---
# def edge_removal_giant_component(G, strategy="weak_to_strong"):
#     G_copy = G.copy()
#
#     if strategy == "weak_to_strong":
#         # Sort edges from weak to strong (ascending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'])
#     elif strategy == "strong_to_weak":
#         # Sort edges from strong to weak (descending order of weight)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
#     elif strategy == "random":
#         # Shuffle edges randomly
#         edges_sorted = list(G_copy.edges(data=True))
#         random.shuffle(edges_sorted)
#     elif strategy == "betweenness":
#         # Sort edges based on betweenness centrality
#         betweenness = nx.edge_betweenness_centrality(G_copy)
#         edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: betweenness[(x[0], x[1])], reverse=True)
#     else:
#         raise ValueError(
#             "Unknown strategy. Choose from 'weak_to_strong', 'strong_to_weak', 'random', or 'betweenness'.")
#
#     giant_sizes = []
#     percents_removed = []
#     num_edges = len(edges_sorted)
#
#     for i, (u, v, d) in enumerate(edges_sorted):
#         G_copy.remove_edge(u, v)
#         if len(G_copy) == 0 or G_copy.number_of_edges() == 0:
#             giant_sizes.append(0)
#         else:
#             giant_sizes.append(len(max(nx.connected_components(G_copy), key=len)))
#         percents_removed.append((i + 1) / num_edges)
#
#     return percents_removed, giant_sizes
#
#
# # --- Plot all strategies on the same graph ---
# def plot_edge_removal_strategies(G):
#     strategies = ['weak_to_strong', 'strong_to_weak', 'random', 'betweenness']
#     colors = ['b', 'g', 'r', 'c']  # Different colors for each strategy
#     labels = ['Weak to Strong', 'Strong to Weak', 'Random', 'Betweenness']
#
#     plt.figure(figsize=(8, 6))
#
#     for i, strategy in enumerate(strategies):
#         x, y = edge_removal_giant_component(G, strategy=strategy)
#         plt.plot(x, y, marker='o', color=colors[i], label=labels[i])
#
#     plt.xlabel('Percentage of edges removed')
#     plt.ylabel('Size of the giant component')
#     plt.title('Edge Removal and Giant Component Size - Network of Thrones')
#     plt.grid()
#     plt.legend()
#
#     plt.show()
#######################################################################


# --- General function to remove edges based on selected strategy ---
def edge_removal_giant_component(G, strategy="weak_to_strong"):
    G_copy = G.copy()

    if strategy == "weak_to_strong":
        # Sort edges from weak to strong (ascending order of weight)
        edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'])
    elif strategy == "strong_to_weak":
        # Sort edges from strong to weak (descending order of weight)
        edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    elif strategy == "random":
        # Shuffle edges randomly
        edges_sorted = list(G_copy.edges(data=True))
        random.shuffle(edges_sorted)
    elif strategy == "betweenness":
        # Sort edges based on betweenness centrality
        betweenness = nx.edge_betweenness_centrality(G_copy)
        edges_sorted = sorted(G_copy.edges(data=True), key=lambda x: betweenness[(x[0], x[1])], reverse=True)
    else:
        raise ValueError(
            "Unknown strategy. Choose from 'weak_to_strong', 'strong_to_weak', 'random', or 'betweenness'.")

    giant_sizes = []
    removed_edges = []  # Track the number of removed edges
    num_edges = len(edges_sorted)

    for i, (u, v, d) in enumerate(edges_sorted):
        G_copy.remove_edge(u, v)
        removed_edges.append(i + 1)  # Count the number of removed edges
        if len(G_copy) == 0 or G_copy.number_of_edges() == 0:
            giant_sizes.append(0)
        else:
            giant_sizes.append(len(max(nx.connected_components(G_copy), key=len)))

    return removed_edges, giant_sizes


# --- Plot all strategies on the same graph ---
def plot_edge_removal_strategies(G):
    strategies = ['weak_to_strong', 'strong_to_weak', 'random', 'betweenness']
    colors = ['b', 'g', 'r', 'c']  # Different colors for each strategy
    labels = ['Weak to Strong', 'Strong to Weak', 'Random', 'Betweenness']

    plt.figure(figsize=(8, 6))

    for i, strategy in enumerate(strategies):
        x, y = edge_removal_giant_component(G, strategy=strategy)
        plt.plot(x, y, marker='o', color=colors[i], label=labels[i])

    plt.xlabel('Number of edges removed')  # Change X axis label
    plt.ylabel('Size of the giant component')
    plt.title('Edge Removal and Giant Component Size - My Network ')
    plt.grid()
    plt.legend()

    plt.show()




######################################################################


# --- Function to compute Neighborhood Overlap ---
def compute_neighborhood_overlap(G):
    overlaps = []
    weights = []

    for u, v, d in G.edges(data=True):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v) - 2  # Subtract u and v themselves
        if union > 0:
            overlap = common / union
            overlaps.append(overlap)
            weights.append(d['weight'])

    return weights, overlaps

def main():

    """**********************************************************************
    **********************************************************************
    Network of Thrones
    **********************************************************************
    *********************************************************************"""
    #--- Reading the files ---
    edges_df = pd.read_csv('stormofswords.csv')  # Edges with weights
    nodes_df = pd.read_csv('tribes.csv', header=None)  # Only nodes
    nodes = nodes_df[0].tolist()

    # --- Building the graph ---
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for idx, row in edges_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])



    """**********************************************************************
      Task 1 :
     *********************************************************************"""
    plot_edge_removal_strategies(G)

    """**********************************************************************
        Task 2 :
     *********************************************************************"""

    # --- Compute for the graph ---
    weights, overlaps = compute_neighborhood_overlap(G)
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.scatter(weights, overlaps, alpha=0.6)
    plt.xlabel('Edge weight')
    plt.ylabel('Neighborhood Overlap')
    plt.title('Neighborhood Overlap as a Function of Weight - Network of Thrones')
    plt.grid()
    plt.show()


    """**********************************************************************
    **********************************************************************
    My Facebook Network - Subgraph of only Politician and TV-show pages
    **********************************************************************
    *********************************************************************"""
    # # Load the edge and target data files
    # edges_df = pd.read_csv('musae_facebook_edges.csv')  # Edge list (relationships between nodes)
    # targets_df = pd.read_csv('musae_facebook_target.csv')  # Node attributes (e.g., page type)
    #
    # # Create the full graph with all edges from the edge list
    # G_full = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')  # Create the graph using the edge list
    #
    # # Identify the types of nodes based on their category - politicians or government organizations
    # politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()  # List of politician nodes
    # government_nodes = targets_df[targets_df['page_type'] == 'tvshow']['id'].tolist()  # List of government organization nodes
    #
    # # Create a filtered graph that only contains the nodes of politicians and government organizations
    # G = G_full.subgraph(politician_nodes + government_nodes)  # Subgraph with only politician and government nodes

    """**********************************************************************
    Find the largest connected component subgraph and Working only on it 
    *********************************************************************"""
    # largest_subgraph = largest_connected_component_subgraph(G)
    """**********************************************************************
      Task 1 :
     *********************************************************************"""
    # Read the weighted graph from the CSV file
    edges_df = pd.read_csv("weighted_graph.csv")

    # Build the graph
    weighted_graph = nx.Graph()
    for idx, row in edges_df.iterrows():
        weighted_graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    plot_edge_removal_strategies(weighted_graph)

    # """**********************************************************************
    #     Task 2 :
    #  *********************************************************************"""

    # --- Compute for the graph ---
    weights, overlaps = compute_neighborhood_overlap(weighted_graph)
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.scatter(weights, overlaps, alpha=0.6)
    plt.xlabel('Edge weight')
    plt.ylabel('Neighborhood Overlap')
    plt.title('Neighborhood Overlap as a Function of Weight - My Network ')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
