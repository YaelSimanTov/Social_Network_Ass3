import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random



# --- Counting the number of edges for each weight ---
def count_edge_weights(G):
    weight_count = {}

    # Iterate through all edges and count the occurrence of each weight
    for u, v, d in G.edges(data=True):
        weight = d['weight']
        if weight in weight_count:
            weight_count[weight] += 1
        else:
            weight_count[weight] = 1

    return weight_count

# --- Function to count the number of edges ---
def count_edges(G):
    return G.number_of_edges()

# --- Function to count the number of connected components ---
def count_connected_components(G):
    return len(list(nx.connected_components(G)))



def calculate_average_degree(G):
    """
    Calculates the average degree of a given undirected graph.

    Parameters:
    G (networkx.Graph): An undirected graph

    Returns:
    float: Average degree of the graph
    """
    if len(G.nodes) == 0:
        return 0.0  # Avoid division by zero
    return sum(dict(G.degree()).values()) / len(G.nodes)


def compute_graph_diameter(graph):
    """
    Computes the diameter of a connected graph.
    If the graph is not connected, it will raise an exception unless it's handled.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        int: Diameter of the graph.
    """
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        print(f"The diameter of the graph is: {diameter}")
        return diameter
    else:
        print("The graph is not connected. Calculating diameter for the largest connected component.")
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        print(f"The diameter of the largest connected component is: {diameter}")
        return diameter

def small_world_property(G):
    # Find the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Calculate the average shortest path length between nodes
    avg_shortest_path_length = nx.average_shortest_path_length(G)

    # Calculate the average clustering coefficient (a measure of clustering between neighbors)
    avg_clustering_coefficient = nx.average_clustering(G)

    return avg_shortest_path_length, avg_clustering_coefficient


# --- Function to get the largest connected component subgraph ---
def largest_connected_component_subgraph(G):
    # Get all connected components in the graph
    components = list(nx.connected_components(G))

    # Find the largest component (by size)
    largest_component = max(components, key=len)

    # Create a subgraph from the largest component
    largest_subgraph = G.subgraph(largest_component).copy()

    return largest_subgraph

def count_nodes_by_category(G, targets_df):
    """
    Receives a graph and a DataFrame with category labels,
    and prints the number of nodes from each category in the graph.
    """
    page_types = ['politician', 'company', 'government', 'tvshow']
    counts = {}
    for page_type in page_types:
        nodes_in_category = targets_df[targets_df['page_type'] == page_type]['id']
        count = len([node for node in nodes_in_category if node in G.nodes()])
        counts[page_type] = count

    for page_type, count in counts.items():
        print(f"Number of {page_type}s: {count}")

def analyze_connected_components(G):
    """
    Receives a graph and prints the number of connected components
    and the components.
    """
    num_components = nx.number_connected_components(G)
    print(f"🔹 Number of connected components in the graph: {num_components}")

    components = list(nx.connected_components(G))
    for i, comp in enumerate(sorted(components, key=len, reverse=True)):
        print(f"Component {i+1}: {len(comp)} nodes")


def main():
    """**********************************************************************
    **********************************************************************
    Network of Thrones
    **********************************************************************
    *********************************************************************"""
    # # --- Reading the files ---
    # edges_df = pd.read_csv('stormofswords.csv')  # Edges with weights
    # nodes_df = pd.read_csv('tribes.csv', header=None)  # Only nodes
    # nodes = nodes_df[0].tolist()
    #
    # # --- Building the graph ---
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # for idx, row in edges_df.iterrows():
    #     G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    #
    # # --- Drawing the graph ---
    # pos = nx.spring_layout(G, seed=42)
    #
    # # Draw the filtered graph
    # plt.figure(figsize=(12, 8))
    # nx.draw(G, pos=pos, with_labels=False, node_size=30, node_color='skyblue')
    # plt.title("Graph of Politician (Red) and TV-show (yellow) Pages")
    # plt.show()
    #

    # """**********************************************************************
    #   Task 0 :
    #  *********************************************************************"""
    # print("average degree of the graph : " , calculate_average_degree(G))
    #
    # compute_graph_diameter(G)
    #
    # # Calculate the small-world property metrics
    # avg_shortest_path_length, avg_clustering_coefficient = small_world_property(G)
    #
    # # Print the results
    # print(f"Average Shortest Path Length: {avg_shortest_path_length:.4f}")
    # print(f"Average Clustering Coefficient: {avg_clustering_coefficient:.4f}")
    #
    #
    # largest_subgraph = largest_connected_component_subgraph(G)
    # # You can display the number of nodes and edges in the largest component subgraph:
    # print(f"Number of nodes in the largest component: {largest_subgraph.number_of_nodes()}")
    # print(f"Number of edges in the largest component: {largest_subgraph.number_of_edges()}")
    # """**********************************************************************
    #     Counting the number of edges for each weight
    #  *********************************************************************"""
    #
    # # --- Get the count of edges for each weight ---
    # weight_count = count_edge_weights(G)
    #
    # # --- Sort the weight_count dictionary by weight (key) ---
    # sorted_weight_count = sorted(weight_count.items())
    #
    # # --- Print the sorted result ---
    # for weight, count in sorted_weight_count:
    #     print(f"Weight: {weight}, Number of edges: {count}")
    #
    # """**********************************************************************
    # Counting the number of edges and connected components
    # *********************************************************************"""
    #
    # num_edges = count_edges(G)
    # num_components = count_connected_components(G)
    #
    # print(f"Number of edges: {num_edges}")
    # print(f"Number of connected components: {num_components}")
    #

    """**********************************************************************
       **********************************************************************
       My Facebook Network - Subgraph of only Politician and TV-show pages
       **********************************************************************
    *********************************************************************"""
    # Load the edge and target data files
    edges_df = pd.read_csv('musae_facebook_edges.csv')  # Edge list (relationships between nodes)
    targets_df = pd.read_csv('musae_facebook_target.csv')  # Node attributes (e.g., page type)

    # Create the full graph with all edges from the edge list
    G_full = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')  # Create the graph using the edge list

    # Identify the types of nodes based on their category - politicians or tvshow
    politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()  # List of politician nodes
    tvshow_nodes = targets_df[targets_df['page_type'] == 'tvshow']['id'].tolist()  # List of tvshow nodes

    # Create a filtered graph that only contains the nodes of politicians and tvshow
    G = G_full.subgraph(politician_nodes + tvshow_nodes)  # Subgraph with only politician and tvshow nodes

    """
    ********************************************************************************************
    Draw the Subgraph of only Politician and TV-show pages
    ********************************************************************************************
    """

    # Assign colors
    node_colors = ['red' if node in politician_nodes else 'yellow' for node in G.nodes]

    # Layout for the subgraph
    pos = nx.spring_layout(G, seed=42)

    # Draw the filtered graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=pos, with_labels=False, node_size=30, node_color=node_colors)
    plt.title("Graph of Politician (Red) and TV-show (yellow) Pages")
    plt.show()

    """**********************************************************************
    Counting the number of edges and connected components
    *********************************************************************"""

    # num_edges = count_edges(G)
    # num_components = count_connected_components(G)
    #
    # print(f"Number of edges: {num_edges}")
    # print(f"Number of nodes: {G.number_of_nodes()}")
    # print(f"Number of connected components: {num_components}")
    #analyze_connected_components(G)

    # count_nodes_by_category(G, targets_df)
    #
    """**********************************************************************
    Find the largest connected component subgraph and Working only on it 
    *********************************************************************"""
    largest_subgraph = largest_connected_component_subgraph(G)
    # # You can display the number of nodes and edges in the largest component subgraph:
    # print(f"Number of nodes in the largest component: {largest_subgraph.number_of_nodes()}")
    # print(f"Number of edges in the largest component: {largest_subgraph.number_of_edges()}")
    #

    # count_nodes_by_category(largest_subgraph, targets_df)

    # print("average degree of the graph : " , calculate_average_degree(largest_subgraph))
    #
    # compute_graph_diameter(largest_subgraph)
    #
    # # Calculate the small-world property metrics
    # avg_shortest_path_length, avg_clustering_coefficient = small_world_property(largest_subgraph)
    #
    # # Print the results
    # print(f"Average Shortest Path Length: {avg_shortest_path_length:.4f}")
    # print(f"Average Clustering Coefficient: {avg_clustering_coefficient:.4f}")
    #

    """**********************************************************************
    Draw the largest connected component subgraph  
    *********************************************************************"""
    #########################################################################
    # node_colors = ['red' if node in politician_nodes else 'yellow' for node in largest_subgraph.nodes]
    #
    # # Layout for the subgraph
    # pos = nx.spring_layout(largest_subgraph, seed=42)
    #
    # # Draw the filtered graph
    # plt.figure(figsize=(12, 8))
    # nx.draw(largest_subgraph, pos=pos, with_labels=False, node_size=30, node_color=node_colors)
    # plt.title("Graph of Politician (Red) and TV-show (yellow) Pages")
    # plt.show()
    #

if __name__ == '__main__':
    main()




