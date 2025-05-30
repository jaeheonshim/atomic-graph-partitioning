from .base import GraphPartitioner
import numpy as np
from .gather_neighbors import descendants_at_distance_multisource, edge_boundary

class GridPartitioner(GraphPartitioner):
    def partition(self, atoms, adj_list, desired_partitions, mp = 1):
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

        x_idx = np.mod(np.digitize(scaled[:, 0], x_bins) - 1, granularity)
        y_idx = np.mod(np.digitize(scaled[:, 1], y_bins) - 1, granularity)
        z_idx = np.mod(np.digitize(scaled[:, 2], z_bins) - 1, granularity)

        part_indices = x_idx * granularity**2 + y_idx * granularity + z_idx

        partitions = [[] for _ in range(granularity**3)]
        for i, part_idx in enumerate(part_indices):
            partitions[part_idx].append(i)

        core_partitions = [set(p) for p in partitions]
        num_partitions = len(core_partitions)

        boundary_nodes = [edge_boundary(adj_list, partitions[i]) for i in range(num_partitions)]

        # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
        extended_neighbors = [set(descendants_at_distance_multisource(adj_list, boundary_nodes[i], distance=mp)) for i in range(num_partitions)]

        extended_partitions = [p.union(a) for p, a in zip(core_partitions, extended_neighbors)]

        return partitions, extended_partitions