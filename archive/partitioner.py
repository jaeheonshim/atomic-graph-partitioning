import networkx as nx
import metis

from collections import deque

def part_graph_extended(G, desired_partitions, distance=None):
    def descendants_at_distance_multisource(G, sources, distance=None):
        if sources in G:
            sources = [sources]

        queue = deque(sources)
        depths = deque([0 for _ in queue])
        visited = set(sources)

        for source in queue:
            if source not in G:
                raise nx.NetworkXError(f"The node {source} is not in the graph.")

        while queue:
            node = queue[0]
            depth = depths[0]

            if distance is not None and depth > distance: return

            yield queue[0]

            queue.popleft()
            depths.popleft()

            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    depths.append(depth + 1)

    _, parts = metis.part_graph(G, desired_partitions, objtype="cut")
    partition_map = {node: parts[i] for i, node in enumerate(G.nodes())}
    num_partitions = desired_partitions

    # Find indices of nodes in each partition
    partitions = [set() for _ in range(desired_partitions)]

    for i, node in enumerate(G.nodes()):
        partitions[partition_map[i]].add(node)

    # Find boundary nodes (vertices adjacent to vertex not in partition)
    boundary_nodes = [set(map(lambda uv: uv[0], nx.edge_boundary(G, partitions[i]))) for i in range(num_partitions)]

    # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
    extended_neighbors = [set(descendants_at_distance_multisource(G, boundary_nodes[i], distance=distance)) for i in range(num_partitions)]

    extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

    return partitions, extended_partitions