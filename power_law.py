import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.stats import linregress


def power_law_binning_logarithm(G, bins=20, show_fit=False):
    import warnings
    warnings.filterwarnings("ignore")

    degrees = [d for _, d in G.degree()]
    min_deg = max(min(degrees), 1)
    max_deg = max(degrees)
    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bins)

    hist, bin_edges = np.histogram(degrees, bins=log_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=np.diff(bin_edges), align='center', alpha=0.7, color='skyblue', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title("Log-binned Degree Distribution")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        try:
            import powerlaw
            fit = powerlaw.Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            print(f"  Log-Binned Fit Results:")
            print(f"  α (power-law exponent): {alpha:.3f}")
            print(f"  xmin: {xmin}")
            print(f"  Distribution is power-law? {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

            x_fit = np.linspace(xmin, max_deg, 100)

            # Normalize the fit line to start from the same y-value as the bar chart
            y_fit = (x_fit / xmin) ** (-alpha)
            y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
            # just if want part of x scale
            plt.xlim(left=xmin)


            plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
            plt.legend()

        except ImportError:
            print("you need to install 'powerlaw' to use in 'fit' function")

    # === Linear regression in log-log space ===

    # Select only bins with non-zero probability for regression
    mask = (bin_centers > 0) & (hist > 0)
    log_x = np.log10(bin_centers[mask])
    log_y = np.log10(hist[mask])

    if len(log_x) >= 2:  # At least two points needed for regression
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

        print("\nLinear Regression (log-log space):")
        print(f"  Estimated power-law exponent (slope): {-slope:.3f}")
        print(f"  R-squared: {r_value ** 2:.3f}")

        # Plot the regression line
        x_fit = np.linspace(log_x.min(), log_x.max(), 100)
        y_fit = intercept + slope * x_fit

        plt.plot(10**x_fit, 10**y_fit, 'g--', label='Linear fit (log-log)')
        plt.legend()
    else:
        print("⚠️ Not enough points for linear regression.")

    plt.tight_layout()
    plt.show()



# =====================
# Degree Distribution Histogram
# =====================

# Load the data
edges_df = pd.read_csv("musae_facebook_edges.csv")
targets_df = pd.read_csv("musae_facebook_target.csv")


# Add 'page_name' to the graph (if available)

# Build the graph
G = nx.Graph()
for node_id, page_type in zip(targets_df['id'], targets_df['page_type']):
    G.add_node(node_id, page_type=page_type)

for u, v in zip(edges_df['id_1'], edges_df['id_2']):
    G.add_edge(u, v)


if 'page_name' in targets_df.columns:
    for node_id, name in zip(targets_df['id'], targets_df['page_name']):
        G.nodes[node_id]['page_name'] = name
else:
    print("Warning: 'page_name' column not found in target CSV.")

power_law_binning_logarithm(G)

# Calculate degrees and sort descending
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]

# Print top 10 nodes with degree, page_type, and page_name
print("Top 10 nodes with highest degree:\n")
for node, degree in top_nodes:
    page_type = G.nodes[node].get('page_type', 'Unknown')
    page_name = G.nodes[node].get('page_name', 'Unknown')
    print(f"Node ID: {node}, Degree: {degree}, Type: {page_type}, Name: {page_name}")


"""
********************************************************************************************
Create subgraphs by category
********************************************************************************************
"""

# ************************* Identify nodes per category  *************************
politician_nodes = targets_df[targets_df['page_type'] == 'politician']['id'].tolist()
government_nodes = targets_df[targets_df['page_type'] == 'government']['id'].tolist()
tvshow_nodes = targets_df[targets_df['page_type'] == 'tvshow']['id'].tolist()
company_nodes = targets_df[targets_df['page_type'] == 'company']['id'].tolist()


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

power_law_binning_logarithm(G_politician)
power_law_binning_logarithm(G_government)
power_law_binning_logarithm(G_tvshow)
power_law_binning_logarithm(G_company)



