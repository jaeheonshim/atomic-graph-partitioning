import networkx as nx
import metis

from collections import deque

def descendants_at_distance_multisource(adj_list, sources, distance=None):
    queue = deque(sources)
    depths = deque([0 for _ in queue])
    visited = set(sources)

    while queue:
        node = queue[0]
        depth = depths[0]

        if distance is not None and depth > distance: return

        yield queue[0]

        queue.popleft()
        depths.popleft()

        for adj in adj_list[node]:
            if adj not in visited:
                visited.add(adj)
                queue.append(adj)
                depths.append(depth + 1)
                
def edge_boundary(adj_list, nbunch1):
    boundary = set()
    
    for u, adj in enumerate(adj_list):
        for v in adj:
            if u in nbunch1 and v not in nbunch1:
                boundary.add(v)
                
    return boundary

def part_graph_extended(adj_list, desired_partitions, distance):
    _, parts = metis.part_graph(adj_list, desired_partitions, objtype="cut", numbering=0)
    partition_map = {node: parts[node] for node in range(len(adj_list))}
    num_partitions = desired_partitions

    # Find indices of nodes in each partition
    partitions = [set() for _ in range(desired_partitions)]

    for node in range(len(adj_list)):
        partitions[partition_map[node]].add(node)

    # Find boundary nodes (vertices adjacent to vertex not in partition)
    boundary_nodes = [edge_boundary(adj_list, partitions[i]) for i in range(num_partitions)]

    # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
    extended_neighbors = [set(descendants_at_distance_multisource(adj_list, boundary_nodes[i], distance=distance)) for i in range(num_partitions)]

    extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

    return partitions, extended_partitions