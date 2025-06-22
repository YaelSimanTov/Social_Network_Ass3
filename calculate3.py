import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import inspect
import pandas as pd


def visualize_top_nodes_by_centrality(G, centrality_func, category_dict, top_n=100, title="Top Nodes by Centrality"):
    """
     Visualizes a graph with the top_n nodes having the highest centrality values according to the provided centrality function.

     Parameters:
     - G: NetworkX graph.
     - centrality_func: A NetworkX centrality function (e.g., nx.betweenness_centrality).
     - category_dict: A dictionary mapping category names to lists of node IDs (e.g., {"politician": [...], "government": [...]}).
     - top_n: Number of top nodes to visualize.
     - title: Title of the plot.
     """

    # Compute centrality values
    centrality = centrality_func(G)

    # Sort and select top N nodes
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
    G_top = G.subgraph(top_nodes)

    # Set node sizes based on centrality
    node_sizes = [centrality[node] * 10000 for node in G_top.nodes]

    # Set node colors based on category
    node_colors = []
    color_map = {
        list(category_dict.keys())[0]: 'red',
        list(category_dict.keys())[1]: 'blue'
    }
    for node in G_top.nodes:
        color_found = False
        for cat, nodes in category_dict.items():
            if node in nodes:
                node_colors.append(color_map[cat])
                color_found = True
                break
        if not color_found:
            node_colors.append('gray')

    # Layout and draw
    pos = nx.spring_layout(G_top, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G_top, pos, with_labels=False, node_size=node_sizes, node_color=node_colors)

    # Legend
    legend_handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in category_dict]
    plt.legend(handles=legend_handles)
    plt.title(title)
    plt.show()

    # Print top 5
    top_5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by centrality:")
    for node, value in top_5:
        print(f"Node {node}: {value:.6f}")

    # Save to file: use the function name if possible
    try:
        centrality_name = centrality_func.__name__
    except AttributeError:
        centrality_name = "centrality"

    output_file = f"{centrality_name}_values.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for node, value in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
            category = "None"
            for cat, nodes in category_dict.items():
                if node in nodes:
                    category = cat
                    break
            f.write(f"{node}\t{value:.6f}\t{category}\n")

    print(f"Centrality values saved to: {output_file}")

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


def analyze_graph(G):
    # Calculate degrees for all nodes
    degrees = [G.degree(n) for n in G.nodes()]

    # Create a histogram of the PDF (relative distribution)
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), density=True, color='blue', edgecolor='black')
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.title("Degree Distribution - PDF")
    plt.xlim(0, 300)  # Limit x-axis to make the graph clearer
    plt.show()

    # Calculate the Average Path Length and Clustering Coefficient for the filtered graph
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"Average Path Length: {avg_path_length}")

        clustering_coeff = nx.average_clustering(G)
        print(f"Average Clustering Coefficient: {clustering_coeff}")
    else:
        print("The graph is not connected, skipping average path length and clustering coefficient calculations.")


def main():

     # Load the edge and target data files
     edges_df = pd.read_csv('musae_facebook_edges.csv')  # Edge list (relationships between nodes)
     targets_df = pd.read_csv('musae_facebook_target.csv')  # Node attributes (e.g., page type)

     # Create the full graph with all edges from the edge list
     G_full = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')  # Create the graph using the edge list

     # Identify the types of nodes based on their category - politicians or government organizations
     politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()  # List of politician nodes
     government_nodes = targets_df[targets_df['page_type'] == 'government'][
          'id'].tolist()  # List of government organization nodes

     # Create a filtered graph that only contains the nodes of politicians and government organizations
     G = G_full.subgraph(politician_nodes + government_nodes)  # Subgraph with only politician and government nodes
     """
     ********************************************************************************************
     Create subgraphs by category
     ********************************************************************************************
     """

     #  Identify nodes per category
     politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()
     government_nodes = targets_df[targets_df['page_type'] == 'government']['id'].tolist()
     tvshow_nodes = targets_df[targets_df['page_type'] == 'tvshow']['id'].tolist()
     company_nodes = targets_df[targets_df['page_type'] == 'company']['id'].tolist()

     # Create subgraphs by category
     G_politician = G_full.subgraph(politician_nodes)
     G_government = G_full.subgraph(government_nodes)
     G_tvshow = G_full.subgraph(tvshow_nodes)
     G_company = G_full.subgraph(company_nodes)


     """
     ********************************************************************************************
     Calculate the small-world property metrics
     ********************************************************************************************
     """
     # Calculate the small-world property metrics
     avg_shortest_path_length, avg_clustering_coefficient = small_world_property(G)

     # Print the results
     print(f"Average Shortest Path Length: {avg_shortest_path_length:.4f}")
     print(f"Average Clustering Coefficient: {avg_clustering_coefficient:.4f}")


     """
     ********************************************************************************************
     Compute centrality values
     ********************************************************************************************
     """
     category_dict = {
          "politician": politician_nodes,
          "government": government_nodes
     }

     print("Visualizing top nodes by betweenness centrality:")
     visualize_top_nodes_by_centrality(G, nx.betweenness_centrality, category_dict, top_n=100, title="Top 100 Nodes by Betweenness Centrality")

     print("Visualizing top nodes by degree centrality:")
     visualize_top_nodes_by_centrality(G, nx.degree_centrality, category_dict, top_n=100,
                                       title="Top 100 Nodes by Degree Centrality")

     print("\nVisualizing top nodes by closeness centrality:")
     visualize_top_nodes_by_centrality(G, nx.closeness_centrality, category_dict, top_n=100,
                                       title="Top 100 Nodes by Closeness Centrality")

     """
     ********************************************************************************************
     Creating a bar chart (histogram) of the degree distribution
     ********************************************************************************************
     """
     analyze_graph(G)
     analyze_graph(G_full)
     analyze_graph(G_politician)
     analyze_graph(G_government)



if __name__ == "__main__":
     main()
