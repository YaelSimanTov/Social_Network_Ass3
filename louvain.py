import networkx as nx
import community as community_louvain
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# קריאת קובצי הקלט
target = pd.read_csv('musae_facebook_target.csv')
edges  = pd.read_csv('musae_facebook_edges.csv')

# יצירת גרף חדש
G = nx.Graph()

# הוספת הצמתים לגרף עם תכונת page_type
for it, cat in zip(target['id'], target['page_type']):
    G.add_node(it, page_type=cat)

# הוספת הקשתות לגרף
for n1, n2 in zip(edges['id_1'], edges['id_2']):
    G.add_edge(n1, n2)
# שלב 1: זיהוי קהילות באמצעות אלגוריתם Louvain
partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'community')

# שלב 2: יצירת טבלת נתונים עם הקהילה וסוג הדף של כל צומת
community_data = []
for node in G.nodes():
    community_data.append({
        'node': node,
        'community': G.nodes[node]['community'],
        'page_type': G.nodes[node].get('page_type', 'Unknown')  # הגנה למקרה שאין page_type
    })

community_df = pd.DataFrame(community_data)

# שלב 3: חישוב טבלת הצלבות בין סוגי דפים לקהילות 5 הגדולות ביותר
top_communities = community_df['community'].value_counts().head(5).index
community_summary = pd.crosstab(
    community_df[community_df['community'].isin(top_communities)]['page_type'],
    community_df[community_df['community'].isin(top_communities)]['community']
)

# שלב 4: הדמיה באמצעות heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(community_summary, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Page Types in Top 5 Communities")
plt.xlabel("Community ID")
plt.ylabel("Page Type")
plt.tight_layout()
plt.show()


# שלב נוסף: בדיקת מספר הקהילות שזוהו
num_communities = len(set(partition.values()))
print(f"Number of communities detected by Louvain algorithm: {num_communities}")
