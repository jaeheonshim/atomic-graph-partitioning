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


import numpy as np
from collections import defaultdict

def build_adjacency_list(edge_indices: np.ndarray, undirected: bool = False):
    """
    Build adjacency list from edge_indices:
      adjacency_list[dst] = {src | there is an edge from src to dst}
      if undirected is True, then also add adjacency_list[src] = {dst | there is an edge from src to dst}
    """
    adjacency_list = defaultdict(set)
    send_indices = edge_indices[0]
    dst_indices = edge_indices[1]

    for src, dst in zip(send_indices, dst_indices):
        adjacency_list[dst].add(src)
        if undirected:
            adjacency_list[src].add(dst)

    return adjacency_list

def multi_source_bfs(
    roots: set[int],
    adjacency_list: dict[int, set[int]],
    num_message_passing: int
):
    """
    Starting from roots, perform BFS on adjacency_list for num_message_passing steps.
    return the set of visited nodes.
    """
    visited = set(roots)
    frontier = set(roots)

    for _ in range(num_message_passing):
        new_frontier = set()
        for node in frontier:
            new_frontier |= adjacency_list[node]
        
        visited |= new_frontier
        frontier = new_frontier

    return visited

def part_completeness_check(
    tol_num_nodes:int, 
    num_message_passing:int,
    partitions:list,
    entended_partitions:list, 
    edge_indices:np.ndarray
):
    """
    Check if each partition is complete and has all neighbors within root nodes' num_message_passing-hops
    Parameters:
    - tol_num_nodes: total number of nodes in the graph
    - num_message_passing: number of message passing hops
    - partitions: list of sets, each set contains the root nodes in a partition
    - entended_partitions: list of sets, each set contains the root nodes and their neighbors within num_message_passing-hops
    - edge_indices: np.ndarray of shape (2, num_edges), each column is an edge (src, dst)
    """
    
    ## check there is no overlapping and missing nodes
    all_nodes = set()
    for i, partition in enumerate(partitions):
        if not partition.isdisjoint(all_nodes):
            print(f"There are overlapping nodes in partition {i}")
            return False
        all_nodes |= partition

    if all_nodes != set(range(tol_num_nodes)):
        print("There are missing or extra nodes in partitions.")
        return False
    
    ## check all num_message_passing-hops neighbors are included
    adjacency_list = build_adjacency_list(edge_indices, undirected=False)
    for i, partition in enumerate(partitions):
        visited = multi_source_bfs(partition, adjacency_list, num_message_passing)
        if visited != entended_partitions[i]:
            print(f"There are missing neighbors in partition {i}.")
            return False
    return True
            