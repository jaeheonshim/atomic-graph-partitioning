import networkx as nx
import numpy as np
from collections import deque

import metis

from sklearn.cluster import SpectralClustering

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

def part_metis(atoms, G, desired_partitions, distance=1) -> tuple[list[set], list[set]]:
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

def part_metis_unweighted(atoms, G, desired_partitions, distance=None) -> tuple[list[set], list[set]]:
    # Make a copy of G with no edge weights
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes())
    G_unweighted.add_edges_from(G.edges())

    # Partition using METIS
    _, parts = metis.part_graph(G_unweighted, desired_partitions, objtype="cut")

    # Map nodes to partitions
    partition_map = {node: parts[i] for i, node in enumerate(G.nodes())}
    partitions = [set() for _ in range(desired_partitions)]
    for i, node in enumerate(G.nodes()):
        partitions[partition_map[i]].add(node)

    # Find boundary nodes
    boundary_nodes = [
        set(u for u, v in nx.edge_boundary(G, part)) for part in partitions
    ]

    # Find extended neighbors from boundary nodes
    extended_neighbors = [
        set(descendants_at_distance_multisource(G, boundary, distance=distance))
        for boundary in boundary_nodes
    ]

    extended_partitions = [
        core.union(ghosts) for core, ghosts in zip(partitions, extended_neighbors)
    ]

    return partitions, extended_partitions

def part_spectral(atoms, G, desired_partitions, distance=None) -> tuple[list[set], list[set]]:
    adj_mat = nx.to_numpy_array(G)
    sc = SpectralClustering(desired_partitions, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(adj_mat)

    # Find indices of nodes in each partition
    partitions = [set() for _ in range(desired_partitions)]

    for i, node in enumerate(G.nodes()):
        partitions[labels[i]].add(node)

    # Find boundary nodes (vertices adjacent to vertex not in partition)
    boundary_nodes = [set(map(lambda uv: uv[0], nx.edge_boundary(G, partitions[i]))) for i in range(desired_partitions)]

    # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
    extended_neighbors = [set(descendants_at_distance_multisource(G, boundary_nodes[i], distance=distance)) for i in range(desired_partitions)]

    extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

    return partitions, extended_partitions

def part_grid(atoms, G, desired_partitions, distance=None):
    positions = atoms.get_scaled_positions(wrap=True)
    num_nodes = len(atoms)

    granularity = int(round(desired_partitions ** (1/3)))
    partitions = [[] for _ in range(granularity ** 3)]

    scaled = np.mod(positions, 1.0)
    
    x_min, y_min, z_min = np.min(scaled, axis=0)
    x_max, y_max, z_max = np.max(scaled, axis=0)

    x_bins = np.linspace(x_min, x_max, granularity + 1)
    y_bins = np.linspace(y_min, y_max, granularity + 1)
    z_bins = np.linspace(z_min, z_max, granularity + 1)

    for i in range(num_nodes):
        x_idx = np.mod(np.digitize(scaled[i][0], x_bins) - 1, granularity)
        y_idx = np.mod(np.digitize(scaled[i][1], y_bins) - 1, granularity)
        z_idx = np.mod(np.digitize(scaled[i][2], z_bins) - 1, granularity)
        part_idx = x_idx * granularity**2 + y_idx * granularity + z_idx
        partitions[part_idx].append(i)

    core_partitions = [set(p) for p in partitions]

    boundary_nodes = [
        set(u for u, v in nx.edge_boundary(G, part)) for part in core_partitions
    ]
    extended_neighbors = [
        set(descendants_at_distance_multisource(G, boundary, distance=distance))
        for boundary in boundary_nodes
    ]
    extended_partitions = [
        core.union(ghosts) for core, ghosts in zip(core_partitions, extended_neighbors)
    ]

    return core_partitions, extended_partitions