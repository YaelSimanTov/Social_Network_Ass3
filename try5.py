import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from make_my_graph_weighted import largest_connected_component_subgraph

# =====================
# Load the Facebook graph
# =====================
edges_df = pd.read_csv("musae_facebook_edges.csv")
targets_df = pd.read_csv("musae_facebook_target.csv")

allowed_types = ['politician', 'tvshow']
filtered_targets = targets_df[targets_df['page_type'].isin(allowed_types)]

G = nx.Graph()
for _, row in filtered_targets.iterrows():
    color = 'red' if row['page_type'] == 'politician' else 'yellow'
    G.add_node(row['id'], type=row['page_type'], color=color)

for _, row in edges_df.iterrows():
    if row['id_1'] in G.nodes and row['id_2'] in G.nodes:
        type1 = G.nodes[row['id_1']]['type']
        type2 = G.nodes[row['id_2']]['type']
        if type1 != type2:
            G.add_edge(row['id_1'], row['id_2'])

G = largest_connected_component_subgraph(G)

# =====================
# Spreading function (until no new nodes)
# =====================
def spread_message_until_done(G, source_nodes, p, regulation_set=None):
    informed = set(source_nodes)
    history = []

    while True:
        new_informed = set(informed)
        for node in informed:
            node_color = G.nodes[node]['color']
            for neighbor in G.neighbors(node):
                if neighbor in informed:
                    continue
                neighbor_color = G.nodes[neighbor]['color']
                prob = p if node_color == neighbor_color else (1 - p)
                if random.random() < prob:
                    new_informed.add(neighbor)

        if regulation_set:
            new_informed |= regulation_set

        if new_informed == informed:  # No new nodes were informed
            break

        informed = new_informed
        reds = sum(1 for n in informed if G.nodes[n]['color'] == 'red')
        yellows = sum(1 for n in informed if G.nodes[n]['color'] == 'yellow')
        history.append((reds, yellows))

    return history

# =====================
# Regulation functions (returning sets)
# =====================
def rlr_set(rho):
    return set(random.sample(list(G.nodes()), int(len(G) * rho)))

def blue_only_set(rho):
    yellow_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'yellow']
    return set(random.sample(yellow_nodes, int(len(yellow_nodes) * rho)))

# =====================
# Averaging Function (dynamic length)
# =====================
def average_spread(G, source_selector, p, regulation_set_generator=None, runs=100):
    all_reds = []
    all_yellows = []
    lengths = []

    for _ in range(runs):
        source_nodes = source_selector()
        reg_set = regulation_set_generator() if regulation_set_generator else None
        history = spread_message_until_done(G, source_nodes, p, regulation_set=reg_set)

        if not history:
            continue  # דלג על הריצה אם לא הייתה הפצה

        reds = [r for r, y in history]
        yellows = [y for r, y in history]
        lengths.append(len(history))

        all_reds.append(reds)
        all_yellows.append(yellows)

    if not all_reds:  # אם כל הריצות נכשלו
        print("No valid spread occurred in any of the runs.")
        return [], []

    max_len = max(len(r) for r in all_reds)

    # Pad with last value
    reds_padded = [r + [r[-1]] * (max_len - len(r)) for r in all_reds]
    yellows_padded = [y + [y[-1]] * (max_len - len(y)) for y in all_yellows]

    reds_avg = np.mean(reds_padded, axis=0)
    yellows_avg = np.mean(yellows_padded, axis=0)

    print(f"Average diffusion length: {np.mean(lengths):.2f} steps")
    return reds_avg, yellows_avg
# =====================
# Source Selector
# =====================
def select_random_reds(k=1):
    red_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'red']
    return random.sample(red_nodes, k)

# =====================
# Run Scenarios
# =====================
scenarios = {
    "Strong No-Reg": (1.0, None),
    "p=0.7 No-Reg": (0.7, None),
    "Strong RLR(0.25)": (1.0, lambda: rlr_set(0.25)),
    "p=0.7 RLR(0.25)": (0.7, lambda: rlr_set(0.25)),
    "Strong BlueOnly(0.25)": (1.0, lambda: blue_only_set(0.25)),
    "p=0.7 BlueOnly(0.25)": (0.7, lambda: blue_only_set(0.25)),
}

average_results = {}
for label, (p_val, reg_fn) in scenarios.items():
    print(f"Running scenario: {label}")
    reds_avg, yellows_avg = average_spread(G, select_random_reds, p=p_val, regulation_set_generator=reg_fn, runs=100)
    average_results[label] = (reds_avg, yellows_avg)

# =====================
# Plot Results
# =====================
plt.figure(figsize=(10, 6))
colors = {
    "p=0.7 No-Reg": "orange",
    "p=0.7 RLR(0.25)": "purple",
    "Strong No-Reg": "blue",
    "Strong RLR(0.25)": "green",
    "Strong BlueOnly(0.25)": "brown",
    "p=0.7 BlueOnly(0.25)": "red"
}

for label, (reds, yellows) in average_results.items():
    plt.plot(reds, label=f"{label} – Red", linestyle='--', color=colors[label])
    plt.plot(yellows, label=f"{label} – Yellow", linestyle='-', color=colors[label])

plt.xlabel("Time Step")
plt.ylabel("Average Informed Users")
plt.title("Average Spread Over 100 Runs (Dynamic Length)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# =====================
# Compute Echo Chamber Metrics
# =====================
print("\n=== Echo Chamber Metrics ===")
for label, (reds_avg, yellows_avg) in average_results.items():
    alpha = reds_avg[-1] + yellows_avg[-1]
    phi_red = reds_avg[-1] / alpha if alpha > 0 else 0
    phi_yellow = yellows_avg[-1] / alpha if alpha > 0 else 0
    print(f"{label}:")
    print(f"  α (size) = {alpha:.2f}, ϕ_red = {phi_red:.2f}, ϕ_yellow = {phi_yellow:.2f}")
