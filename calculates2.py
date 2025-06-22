import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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


def get_min_value_from_centrality(filename):
    """
    Reads a text file with at least two columns, and returns
    the minimum value from the second column.

    Parameters:
        filename (str): Path to the text file.

    Returns:
        float: The minimum value found in the second column.
    """
    # Read the file into a DataFrame, assuming space-separated values and no header
    df = pd.read_csv(filename, sep=r"\s+", header=None, engine='python')

    # Extract and return the minimum value in the second column (index 1)
    return df[1].min()


# Example usage:
if __name__ == "__main__":
    # Load edge and target data
    edges = pd.read_csv("musae_facebook_edges.csv")
    targets = pd.read_csv("musae_facebook_target.csv")

    # Build the graph
    G = nx.from_pandas_edgelist(edges, "id_1", "id_2")

    # Analyze and print results
    analyze_connected_components(G)
    count_nodes_by_category(G, targets)

    #**********************************************************************************************************
    # Identify the types of nodes based on their category - politicians or government organizations
    politician_nodes = targets[targets['page_type'] == 'politician']['id'].tolist()  # List of politician nodes
    government_nodes = targets[targets['page_type'] == 'government'][
        'id'].tolist()  # List of government organization nodes

    # Create a filtered graph that only contains the nodes of politicians and government organizations
    G_filter = G.subgraph(politician_nodes + government_nodes)  # Subgraph with only politician and government nodes
    analyze_connected_components(G_filter)
    count_nodes_by_category(G_filter, targets)

    #**********************************************************************************************************
    avg_deg = calculate_average_degree(G)
    print(f"Average Degree of the graph: {avg_deg:.6f}")

    #*********************************************************************************************************
    diameter = compute_graph_diameter(G)
    print("Graph Diameter:", diameter)

    #**********************************************************************************************************

    # min_val = get_min_value_from_centrality("degree_centrality_values.txt")
    # print("Minimum value in column 2:", min_val)
    #
    #
    # min_val = get_min_value_from_centrality("beetweenness_centrality_values.txt")
    # print("Minimum value in column 2:", min_val)
    #
    # min_val = get_min_value_from_centrality("closeness_centrality_values.txt")
    # print("Minimum value in column 2:", min_val)

