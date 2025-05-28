from gather_neighbors import nx_gather
import metis
import ase
import networkx as nx

def partition(atoms: ase.Atoms, G, desired_partitions, mp=1) -> tuple[list[set], list[set]]:
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
    extended_neighbors = [set(nx_gather(G, boundary_nodes[i], distance=mp)) for i in range(num_partitions)]

    extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

    return partitions, extended_partitions