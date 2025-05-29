from .base import GraphPartitioner
import metis
from collections import deque
from .gather_neighbors import edge_boundary, descendants_at_distance_multisource

class MetisPartitioner(GraphPartitioner):
    def partition(self, atoms, adj_list, desired_partitions, mp = 1):
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
        extended_neighbors = [set(descendants_at_distance_multisource(adj_list, boundary_nodes[i], distance=mp)) for i in range(num_partitions)]

        extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

        return partitions, extended_partitions