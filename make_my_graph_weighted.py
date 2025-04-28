import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- Step 3: Build the weighted graph based on feature overlap ---
def build_weighted_graph(original_graph, node_to_features):
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(original_graph.nodes())

    for u, v in original_graph.edges():
        features_u = node_to_features.get(u, set())
        features_v = node_to_features.get(v, set())
        common_features = features_u & features_v
        weight = len(common_features)

        if weight > 0:  # רק אם יש משקל חיובי נוצרת קשת
            G_weighted.add_edge(u, v, weight=weight)

    return G_weighted


# --- Function to get the largest connected component subgraph ---
def largest_connected_component_subgraph(G):
    # Get all connected components in the graph
    components = list(nx.connected_components(G))

    # Find the largest component (by size)
    largest_component = max(components, key=len)

    # Create a subgraph from the largest component
    largest_subgraph = G.subgraph(largest_component).copy()

    return largest_subgraph

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


# Load the edge and target data files
edges_df = pd.read_csv('musae_facebook_edges.csv')  # Edge list (relationships between nodes)
targets_df = pd.read_csv('musae_facebook_target.csv')  # Node attributes (e.g., page type)

# Create the full graph with all edges from the edge list
G_full = nx.from_pandas_edgelist(edges_df, source='id_1', target='id_2')  # Create the graph using the edge list

# Identify the types of nodes based on their category - politicians or government organizations
politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()  # List of politician nodes
tvshow_nodes = targets_df[targets_df['page_type'] == 'tvshow'][ 'id'].tolist()  # List of government organization nodes

# Create a filtered graph that only contains the nodes of politicians and government organizations
G = G_full.subgraph(politician_nodes + tvshow_nodes)  # Subgraph with only politician and government nodes

largest_subgraph = largest_connected_component_subgraph(G)

# --- Step 1: Load the features ---
features_df = pd.read_csv('filtered_musae_facebook_features.csv')  # קובץ שכבר סיננתי קודם

# --- Step 2: Build a dictionary: node_id -> set of feature_ids ---
node_to_features = {}
for _, row in features_df.iterrows():
    node = row['node_id']
    feature = row['feature_id']
    if node not in node_to_features:
        node_to_features[node] = set()
    node_to_features[node].add(feature)

"""**********************************************************************
saves the node IDs of the subgraph to a file, filters the features to include only 
those nodes, and saves the filtered features to a new file.
**********************************************************************"""
# Load the features file
features_df = pd.read_csv('musae_facebook_features.csv')
subgraph_nodes = list(largest_subgraph.nodes()) # Get the node ids from my subgraph

# Save them into a CSV file
subgraph_nodes_df = pd.DataFrame(subgraph_nodes, columns=['node_id'])
subgraph_nodes_df.to_csv('subgraph_node_ids.csv', index=False)

print("The node IDs have been saved to 'subgraph_node_ids.csv'.")

# Filter the dataframe: keep only rows where node_id is in the subgraph
filtered_features_df = features_df[features_df['node_id'].isin(subgraph_nodes)]
filtered_features_df.to_csv('filtered_musae_facebook_features.csv', index=False)
print(filtered_features_df.head())


"""**********************************************************************
Build the weighted graph
**********************************************************************"""


weighted_graph = build_weighted_graph(largest_subgraph, node_to_features)
# --- Step 4: Save the weighted graph if you want ---
# Save the weighted graph as a CSV file with Source, Target, and Weight columns
edges = []
for u, v, data in weighted_graph.edges(data=True):
    edges.append((u, v, data['weight']))

edges_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
edges_df.to_csv('weighted_graph.csv', index=False)

# --- Step 5: (Optional) Check basic properties ---
print(f"Number of nodes: {weighted_graph.number_of_nodes()}")
print(f"Number of edges: {weighted_graph.number_of_edges()}")


"""**********************************************************************
Draw the graph
**********************************************************************"""
# Assign colors
# node_colors = ['red' if node in politician_nodes else 'yellow' for node in weighted_graph.nodes]
#
# # Layout for the subgraph
# pos = nx.spring_layout(weighted_graph, seed=42)
#
# # Draw the filtered graph
# plt.figure(figsize=(12, 8))
# nx.draw(weighted_graph, pos=pos, with_labels=False, node_size=30, node_color=node_colors)
# plt.title("Graph of Politician (Red) and TV-show (yellow) Pages")
# plt.show()

"""**********************************************************************
Counting the number of edges for each weight -optional $$$ i do not use it yet $$$
*********************************************************************"""

# --- Get the count of edges for each weight ---
weight_count = count_edge_weights(weighted_graph)

# --- Sort the weight_count dictionary by weight (key) ---
sorted_weight_count = sorted(weight_count.items())

# --- Print the sorted result ---
for weight, count in sorted_weight_count:
    print(f"Weight: {weight}, Number of edges: {count}")




