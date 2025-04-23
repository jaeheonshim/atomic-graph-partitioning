from wrapper import partitioner
from partitioning import part_metis_unweighted
import networkx as nx

graph = [
    [1, 2, 5],          # Node 0 connected to 1, 2, 5
    [0, 2, 3, 4],       # Node 1 connected to 0, 2, 3, 4
    [0, 1, 3, 5, 6],    # Node 2 connected to 0, 1, 3, 5, 6
    [1, 2, 4, 6, 7],    # Node 3 connected to 1, 2, 4, 6, 7
    [1, 3, 7],          # Node 4 connected to 1, 3, 7
    [0, 2, 6],          # Node 5 connected to 0, 2, 6
    [2, 3, 5, 7],       # Node 6 connected to 2, 3, 5, 7
    [3, 4, 6]           # Node 7 connected to 3, 4, 6
]

G = nx.Graph()

G.add_nodes_from(range(len(graph)))
for u, neighbors in enumerate(graph):
    for v in neighbors:
        G.add_edge(u, v)

G = nx.Graph(G)

original_partitions = part_metis_unweighted(None, G, 2, distance=1)
wrapper_partitions = partitioner.part_graph_kway_extended(graph, 2, distance=1)

print(original_partitions)
print(wrapper_partitions)