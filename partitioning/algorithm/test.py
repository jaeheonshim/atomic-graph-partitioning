import partitioner

# Create a simple graph (adjacency list)
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

# 0 1 2 3 4 5 6 7 8 9
# 1 2 0 2 3 0 1 3 1 2

# Partition graph into 2 parts
partitions = partitioner.part_graph_kway_extended(graph, 2)
print(partitions)  # Something like [0, 0, 1, 1]