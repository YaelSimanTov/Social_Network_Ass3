import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

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

# הדפסת מספר הצמתים והקשתות (בדיקת sanity)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# מחזיר את הקשת עם ערך מרכזיות הקשת (EBC) הגבוה ביותר
def max_ebc_edge(graph):
    ebc = nx.edge_betweenness_centrality(graph)
    return max(ebc, key=lambda z: ebc[z])

# אלגוריתם Girvan-Newman לפירוק קהילות
def girvan_newman(graph):
    cc = nx.connected_components(graph)
    cnt = nx.number_connected_components(graph)
    num_comm = 1
    while cnt <= num_comm:
        graph.remove_edge(*max_ebc_edge(graph))
        cc = nx.connected_components(graph)
        cnt = nx.number_connected_components(graph)
    return cc

# בדיקת האלגוריתם על גרף פשוט שמובנה ב-NetworkX
F = nx.davis_southern_women_graph()

# הצגת הקשת עם ה-EBC הגבוה ביותר
print("Max EBC edge:", max_ebc_edge(F))

# הפעלת האלגוריתם
c = girvan_newman(F.copy())

# הפיכת הקהילות לרשימה של רשימות
c_nodes = [list(i) for i in c]

# הדפסת מספר הצמתים בכל קהילה
for n in c_nodes:
    print("Community size:", len(n))

# שרטוט הקהילות בצבעים שונים
colors = ['red', 'blue']
color_map = [colors[0] if n in c_nodes[0] else colors[1] for n in F]
nx.draw(F, node_color=color_map, with_labels=True)
plt.show()

# הרצת האלגוריתם על הגרף הגדול (facebook)
gcc = girvan_newman(G.copy())

# הדפסת הקהילות הראשונות בגרף הגדול
gcc_nodes = [list(i) for i in gcc]
for idx, group in enumerate(gcc_nodes):
    print(f"Community {idx + 1} size:", len(group))
