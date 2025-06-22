import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

"""
********************************************************************************************
Load data and Build the full graph
********************************************************************************************
"""
edges = pd.read_csv('musae_facebook_edges.csv')
targets = pd.read_csv('musae_facebook_target.csv')
features = pd.read_csv('musae_facebook_features.csv')

G = nx.from_pandas_edgelist(edges, 'id_1', 'id_2')


"""
********************************************************************************************
Draw the full graph (no categories) 
Uncomment this block to visualize the entire graph without color by category
********************************************************************************************
"""

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # פריסת הצמתים
nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.5, with_labels=False)
plt.title("Facebook Network Graph")
plt.show()


"""
********************************************************************************************
Create subgraphs by category
********************************************************************************************
"""

# ************************* Identify nodes per category  *************************
politician_nodes = targets[targets['page_type'] == 'politician']['id'].tolist()
government_nodes = targets[targets['page_type'] == 'government']['id'].tolist()
tvshow_nodes = targets[targets['page_type'] == 'tvshow']['id'].tolist()
company_nodes = targets[targets['page_type'] == 'company']['id'].tolist()


# ************************* Create subgraphs by category  *************************
G_politician = G.subgraph(politician_nodes)
G_government = G.subgraph(government_nodes)
G_tvshow = G.subgraph(tvshow_nodes)
G_company = G.subgraph(company_nodes)

# Layout formerly each subgraph (fixed for consistency)
pos_politician = nx.spring_layout(G_politician, seed=42)
pos_government = nx.spring_layout(G_government, seed=42)
pos_tvshow = nx.spring_layout(G_tvshow, seed=42)
pos_company = nx.spring_layout(G_company, seed=42)


# ************************* Draw each subgraph separately *************************
# Politician
plt.figure(figsize=(12, 8))
nx.draw(G_politician, pos=pos_politician, with_labels=False, node_size=30, node_color='red')
plt.title("Political Pages Subgraph (Red for Politicians)")
plt.show()

# Government
plt.figure(figsize=(12, 8))
nx.draw(G_government, pos=pos_government, with_labels=False, node_size=30, node_color='blue')
plt.title("Government Pages Subgraph (Blue for Government)")
plt.show()

# TV Show
plt.figure(figsize=(12, 8))
nx.draw(G_tvshow, pos=pos_tvshow, with_labels=False, node_size=30, node_color='yellow')
plt.title("TV Show Pages Subgraph (Yellow for TV Shows)")
plt.show()

# Company
plt.figure(figsize=(12, 8))
nx.draw(G_company, pos=pos_company, with_labels=False, node_size=30, node_color='green')
plt.title("Company Pages Subgraph (Green for Companies)")
plt.show()


"""
********************************************************************************************
Draw the full graph with categories
********************************************************************************************
"""
# Function to assign color based on category
def get_node_color(node):
    if node in politician_nodes:
        return 'red'
    elif node in government_nodes:
        return 'blue'
    elif node in tvshow_nodes:
        return 'yellow'
    elif node in company_nodes:
        return 'green'
    else:
        return 'gray'

# Color list for each node
node_colors = [get_node_color(node) for node in G.nodes]

# Layout for the full graph
pos = nx.spring_layout(G, seed=42)

# Draw the unified graph
plt.figure(figsize=(16, 12))
nx.draw(G, pos=pos, with_labels=False, node_size=20, node_color=node_colors)
plt.title("Unified Facebook Graph - Node Colors by Category", fontsize=16)
plt.show()

"""
********************************************************************************************
Subgraph of only Politician and Government pages
********************************************************************************************
"""
# Combine relevant nodes
relevant_nodes = set(politician_nodes + government_nodes)

# Create subgraph containing only politician and government pages
G_relevant = G.subgraph(relevant_nodes)

# Assign colors
node_colors = ['red' if node in politician_nodes else 'blue' for node in G_relevant.nodes]

# Layout for the subgraph
pos = nx.spring_layout(G_relevant, seed=42)

# Draw the filtered graph
plt.figure(figsize=(12, 8))
nx.draw(G_relevant, pos=pos, with_labels=False, node_size=30, node_color=node_colors)
plt.title("Graph of Politician (Red) and Government (Blue) Pages")
plt.show()




