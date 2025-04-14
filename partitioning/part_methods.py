"""
In this script, we implement and test various partitioning methods for materials structrues.
"""

from __future__ import annotations

import time
import copy
from collections import defaultdict, deque
import rich
import os

from ase import Atoms
from ase.io import read, write
from pymatgen.optimization.neighbors import find_points_in_spheres
import torch
import numpy as np
import metis
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from mattersim.datasets.utils.convertor import GraphConvertor

mattersim_converter = GraphConvertor("m3gnet", 5.0, True, 4.0)

def get_connectivity(atoms: Atoms, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    graph = ase_atoms_to_atom_graphs(atoms)
    senders = graph.senders
    receivers = graph.receivers
    edge_feats = graph.edge_features

    src_indices = senders
    dst_indices = receivers
    
    edge_indices = np.vstack((src_indices, dst_indices))
    edge_lengths = edge_feats
    return edge_indices, edge_lengths

def BFS_extension(
    num_nodes: int,
    edge_indices: np.ndarray,
    partitions: list[list[int]],
    mp_steps: int,
) -> list[list[int]]:
    
    def descendants_at_distance_multisource(G, sources, mp_steps=1):
        if isinstance(sources, int):
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
    
            if mp_steps is not None and depth > mp_steps:
                return
    
            yield queue[0]
    
            queue.popleft()
            depths.popleft()
    
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    depths.append(depth + 1)
    
    num_partitions = len(partitions)     
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    edges = [(src, dst) for src, dst in zip(src_indices, dst_indices)]
    graph.add_edges_from(edges)
    
    # Find boundary nodes (vertices adjacent to vertex not in partition)
    boundary_nodes = [
        set(map(lambda uv: uv[0], nx.edge_boundary(graph, partitions[i])))
        for i in range(num_partitions)
    ]
    # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
    extended_neighbors = [
        set(descendants_at_distance_multisource(graph, boundary_nodes[i], mp_steps=mp_steps))
        for i in range(num_partitions)
    ]
    extended_partitions = [
        list(set(p).union(a))
        for p, a in zip(partitions, extended_neighbors)
    ]
    return extended_partitions

def new_BFS_extension(
    num_nodes: int,
    edge_indices: np.ndarray,
    partitions: list[list[int]],
    mp_steps: int,
) -> list[list[int]]:
    adjancency = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_indices[0], edge_indices[1]):
        adjancency[src].append(dst)
    
    entended_partitions = copy.deepcopy(partitions)
    for i, partition in enumerate(partitions):
        visited = set(partition)
        queue = deque(partition)
        depth = 0
        while queue:
            if depth >= mp_steps:
                break
            for _ in range(len(queue)):
                node = queue.popleft()
                for neighbor in adjancency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        entended_partitions[i].append(neighbor)
            depth += 1
    
    return entended_partitions
    

def partition_atoms(
    atoms: Atoms,
    extended_partitions: list[list[int]],
) -> list[Atoms]:
    partitioned_atoms = []
    for partition in extended_partitions:
        positions = np.array(atoms.get_positions())[partition]
        atomic_numbers = np.array(atoms.get_atomic_numbers())[partition]
        cell = np.array(atoms.get_cell())
        part_atoms = Atoms(
            symbols=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=atoms.pbc,
        )
        partitioned_atoms.append(part_atoms)
    return partitioned_atoms
        

def metis_partition(
    num_nodes: int,
    edge_indices: np.ndarray,
    num_partitions: int,
) -> list[list[int]]:
    # Create a graph from the edge indices and lengths
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    edges = [(src, dst) for src, dst in zip(src_indices, dst_indices)]
    graph.add_edges_from(edges)
    
    # Use METIS to partition the graph
    _, partition_indices = metis.part_graph(graph, nparts=num_partitions, objtype="cut")
    partition_indices = np.array(partition_indices).reshape(-1)
    partitions = []
    for i in range(num_partitions):
        partitions.append(np.where(partition_indices == i)[0].tolist())
    return partitions

def louvain_partition(
    num_nodes: int,
    edge_indices: np.ndarray,
    num_partitions: int,
) -> list[list[int]]:
    graph = nx.Graph()
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    edges = [(src, dst) for src, dst in zip(src_indices, dst_indices)]
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(num_nodes))

    partition_dict = community_louvain.best_partition(graph)
    communities = {}
    for node, comm_id in partition_dict.items():
        communities.setdefault(comm_id, []).append(node)
    communities = list(communities.values())
    
    # Adjust the number of partitions to the target number
    if len(communities) == num_partitions:
        return communities
    elif len(communities) > num_partitions:
        # Merge communities if there are too many
        communities.sort(key=len)  # 按社区大小升序排序
        while len(communities) > num_partitions:
            comm1 = communities.pop(0)
            comm2 = communities.pop(0)
            merged = comm1 + comm2
            communities.append(merged)
            communities.sort(key=len)
        return communities
    else:
        # Divide communities if there are too few
        communities.sort(key=len, reverse=True)
        while len(communities) < num_partitions:
            for i, comm in enumerate(communities):
                if len(comm) > 1:
                    comm_sorted = sorted(comm)
                    mid = len(comm_sorted) // 2
                    part1 = comm_sorted[:mid]
                    part2 = comm_sorted[mid:]
                    communities.pop(i)
                    communities.append(part1)
                    communities.append(part2)
                    break
            else:
                break
            communities.sort(key=len, reverse=True)
        while len(communities) < num_partitions:
            communities.append([])
        return communities
    
def ldg_partition(
    num_nodes: int,
    edge_indices: np.ndarray,
    num_partitions: int,
) -> list[list[int]]:
    capacity = int(np.ceil(num_nodes / num_partitions))

    assignments = np.full(num_nodes, -1, dtype=int)
    partitions = [[] for _ in range(num_partitions)]
    
    neighbors = {i: set() for i in range(num_nodes)}
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    for u, v in zip(src_indices, dst_indices):
        neighbors[u].add(v)
    
    for node in range(num_nodes):
        best_score = -float("inf")
        best_partition = None
        for p in range(num_partitions):
            if len(partitions[p]) >= capacity:
                continue
            neighbor_count = 0
            for nbr in neighbors[node]:
                if assignments[nbr] == p:
                    neighbor_count += 1
            score = neighbor_count + (capacity - len(partitions[p])) / capacity
            if score > best_score:
                best_score = score
                best_partition = p
        if best_partition is None:
            best_partition = 0
        assignments[node] = best_partition
        partitions[best_partition].append(node)
        
    return partitions

def spectral_partition(
    num_nodes: int,
    edge_indices: np.ndarray,
    num_partitions: int,
) -> list[list[int]]:
    graph = nx.Graph()
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    edges = [(src, dst) for src, dst in zip(src_indices, dst_indices)]
    graph.add_edges_from(edges)
    graph.add_nodes_from(range(num_nodes))
    
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=range(num_nodes))
    
    sc = SpectralClustering(
        n_clusters=num_partitions,
        affinity="precomputed",
        random_state=42 
    )
    labels = sc.fit_predict(adjacency_matrix)
    
    partitions = []
    for i in range(num_partitions):
        partitions.append(np.where(labels == i)[0].tolist())
        
    return partitions

def grid_partition(
    num_nodes: int,
    scaled_positions: np.ndarray,
    granularity: int,
):
    partitions = [[] for _ in range(granularity ** 3)]
    scaled_positions = np.mod(scaled_positions, 1.0)
    
    x_min, y_min, z_min = np.min(scaled_positions, axis=0)
    x_max, y_max, z_max = np.max(scaled_positions, axis=0)
    
    x_bins = np.linspace(x_min, x_max, granularity + 1)
    y_bins = np.linspace(y_min, y_max, granularity + 1)
    z_bins = np.linspace(z_min, z_max, granularity + 1)
    
    for i in range(num_nodes):
        x_idx = np.mod(np.digitize(scaled_positions[i][0], x_bins) - 1, granularity)
        y_idx = np.mod(np.digitize(scaled_positions[i][1], y_bins) - 1, granularity)
        z_idx = np.mod(np.digitize(scaled_positions[i][2], z_bins) - 1, granularity)
        partition_index = x_idx * granularity ** 2 + y_idx * granularity + z_idx
        partitions[partition_index].append(i)
        
    return partitions


def main(args_dict: dict):
    atoms_list:list[Atoms] = read(args_dict["structs"], index=":") # type: ignore
    granularity = args_dict["granularity"]
    num_partitions = granularity ** 3
    mp_steps = args_dict["mp_steps"]
    system_name = os.path.basename(args_dict["structs"]).split(".")[0]
    os.makedirs("./part_methods_results", exist_ok=True)
    struct_dir = f"./part_methods_results/{system_name}_structs"
    figures_dir = f"./part_methods_results/{system_name}_figures"
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    wandb.init(project="partition_methods", name=system_name, config=args_dict)
    
    
    conn_times = []
    metis_times = []
    metis_sizes_part = []
    metis_sizes_ext = []
    metis_bfs_times = []
    louvain_times = []
    louvain_sizes_part = []
    louvain_sizes_ext = []
    louvain_bfs_times = []
    ldg_times = []
    ldg_sizes_part = []
    ldg_sizes_ext = []
    ldg_bfs_times = []
    spectral_times = []
    spectral_sizes_part = []
    spectral_sizes_ext = []
    spectral_bfs_times = []
    grid_times = []
    grid_sizes_part = []
    grid_sizes_ext = []
    grid_bfs_times = []
    
    metis_failed = False
    louvain_failed = False
    ldg_failed = False
    spectral_failed = False
    grid_failed = False
    for i, atoms in enumerate(tqdm(atoms_list)):
        time1 = time.time()
        edge_indices, edge_lengths = get_connectivity(atoms, cutoff=5.0)
        edge_indices = edge_indices.astype(np.int32)
        conn_times.append(time.time() - time1)
        num_nodes = len(atoms)
        
        try:
            time1 = time.time()
            partitions = metis_partition(num_nodes, edge_indices, num_partitions)
            time2 = time.time()
            extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
            metis_bfs_times.append(time.time() - time2)
            metis_times.append(time.time() - time1)
            parted_atoms = partition_atoms(atoms, extended_partitions)
            write(f"{struct_dir}/metis_parts_{i}.xyz", parted_atoms)
            metis_sizes_part.extend([len(partition) for partition in partitions])
            metis_sizes_ext.extend([len(partition) for partition in extended_partitions])
            tqdm.write("metis finished")
        except Exception as e:
            tqdm.write(f"metis failed: {e}")
            exit()
            metis_times.append(0)
            metis_bfs_times.append(0)
            metis_sizes_part.append(0)
            metis_sizes_ext.append(0)
        
        try:
            if num_nodes > 24000:
                raise Exception("Too many nodes for spectral clustering")
            louvain_failed = True
            time1 = time.time()
            partitions = louvain_partition(num_nodes, edge_indices, num_partitions)
            time2 = time.time()
            extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
            louvain_bfs_times.append(time.time() - time2)
            louvain_times.append(time.time() - time1)
            parted_atoms = partition_atoms(atoms, extended_partitions)
            write(f"{struct_dir}/louvain_parts_{i}.xyz", parted_atoms)
            louvain_sizes_part.extend([len(partition) for partition in partitions])
            louvain_sizes_ext.extend([len(partition) for partition in extended_partitions])
            tqdm.write("louvain finished")
        except Exception as e:
            tqdm.write(f"louvain failed: {e}")
            louvain_times.append(0)
            louvain_bfs_times.append(0)
            louvain_sizes_part.append(0)
            louvain_sizes_ext.append(0)
        
        try:
            time1 = time.time()
            partitions = ldg_partition(num_nodes, edge_indices, num_partitions)
            time2 = time.time()
            extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
            ldg_bfs_times.append(time.time() - time2)
            ldg_times.append(time.time() - time1)
            parted_atoms = partition_atoms(atoms, extended_partitions)
            write(f"{struct_dir}/ldg_parts_{i}.xyz", parted_atoms)
            ldg_sizes_part.extend([len(partition) for partition in partitions])
            ldg_sizes_ext.extend([len(partition) for partition in extended_partitions])
            tqdm.write("ldg finished")
        except Exception as e:
            tqdm.write(f"ldg failed: {e}")
            ldg_times.append(0)
            ldg_bfs_times.append(0)
            ldg_sizes_part.append(0)
            ldg_sizes_ext.append(0)
        
        try:
            if num_nodes > 24000:
                raise Exception("Too many nodes for spectral clustering")
            spectral_failed = True
            time1 = time.time()
            partitions = spectral_partition(num_nodes, edge_indices, num_partitions)
            time2 = time.time()
            extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
            spectral_bfs_times.append(time.time() - time2)
            spectral_times.append(time.time() - time1)
            parted_atoms = partition_atoms(atoms, extended_partitions)
            write(f"{struct_dir}/spectral_parts_{i}.xyz", parted_atoms)
            spectral_sizes_part.extend([len(partition) for partition in partitions])
            spectral_sizes_ext.extend([len(partition) for partition in extended_partitions])
            tqdm.write("spectral finished")
        except Exception as e:
            tqdm.write(f"spectral failed: {e}")
            spectral_times.append(0)
            spectral_bfs_times.append(0)
            spectral_sizes_part.append(0)
            spectral_sizes_ext.append(0)
        
        try:
            time1 = time.time()
            scaled_positions = np.array(atoms.get_scaled_positions())
            partitions = grid_partition(num_nodes, scaled_positions, granularity)
            time2 = time.time()
            extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
            grid_bfs_times.append(time.time() - time2)
            grid_times.append(time.time() - time1)
            parted_atoms = partition_atoms(atoms, extended_partitions)
            write(f"{struct_dir}/grid_parts_{i}.xyz", parted_atoms)
            grid_sizes_part.extend([len(partition) for partition in partitions])
            grid_sizes_ext.extend([len(partition) for partition in extended_partitions])
            tqdm.write("grid finished")
        except Exception as e:
            tqdm.write(f"grid failed: {e}")
            grid_times.append(0)
            grid_bfs_times.append(0)
            grid_sizes_part.append(0)
            grid_sizes_ext.append(0)
        

    rich.print("Connectivity Time: ", np.mean(conn_times))
    rich.print("METIS Partition Time: ", np.mean(metis_times))
    rich.print("METIS Partition Sizes: ", np.mean(metis_sizes_part))
    rich.print("METIS Extended Partition Sizes: ", np.mean(metis_sizes_ext))
    rich.print("METIS BFS Time: ", np.mean(metis_bfs_times))
    rich.print("Louvain Partition Time: ", np.mean(louvain_times))
    rich.print("Louvain Partition Sizes: ", np.mean(louvain_sizes_part))
    rich.print("Louvain Extended Partition Sizes: ", np.mean(louvain_sizes_ext))
    rich.print("Louvain BFS Time: ", np.mean(louvain_bfs_times))
    rich.print("LDG Partition Time: ", np.mean(ldg_times))
    rich.print("LDG Partition Sizes: ", np.mean(ldg_sizes_part))
    rich.print("LDG Extended Partition Sizes: ", np.mean(ldg_sizes_ext))
    rich.print("LDG BFS Time: ", np.mean(ldg_bfs_times))
    rich.print("Spectral Partition Time: ", np.mean(spectral_times))
    rich.print("Spectral Partition Sizes: ", np.mean(spectral_sizes_part))
    rich.print("Spectral Extended Partition Sizes: ", np.mean(spectral_sizes_ext))
    rich.print("Spectral BFS Time: ", np.mean(spectral_bfs_times))
    rich.print("Grid Partition Time: ", np.mean(grid_times))
    rich.print("Grid Partition Sizes: ", np.mean(grid_sizes_part))
    rich.print("Grid Extended Partition Sizes: ", np.mean(grid_sizes_ext))
    rich.print("Grid BFS Time: ", np.mean(grid_bfs_times))
    
    ## part_size_plot
    plt.figure(figsize=(10, 6))
    plt.hist(metis_sizes_part, bins=20, alpha=0.5, label="METIS")
    plt.hist(louvain_sizes_part, bins=20, alpha=0.5, label="Louvain")
    plt.hist(ldg_sizes_part, bins=20, alpha=0.5, label="LDG")
    plt.hist(spectral_sizes_part, bins=20, alpha=0.5, label="Spectral")
    plt.hist(grid_sizes_part, bins=20, alpha=0.5, label="Grid")
    plt.xlabel("Partition Size")
    plt.ylabel("Frequency")
    plt.title("Partition Size Distribution")
    plt.legend()
    plt.savefig(f"{figures_dir}/partition_size_distribution.png")
    
    ## ext_part_size_plot
    plt.figure(figsize=(10, 6))
    plt.hist(metis_sizes_ext, bins=20, alpha=0.5, label="METIS")
    plt.hist(louvain_sizes_ext, bins=20, alpha=0.5, label="Louvain")
    plt.hist(ldg_sizes_ext, bins=20, alpha=0.5, label="LDG")
    plt.hist(spectral_sizes_ext, bins=20, alpha=0.5, label="Spectral")
    plt.hist(grid_sizes_ext, bins=20, alpha=0.5, label="Grid")
    plt.xlabel("Extended Partition Size")
    plt.ylabel("Frequency")
    plt.title("Extended Partition Size Distribution")
    plt.legend()
    plt.savefig(f"{figures_dir}/extended_partition_size_distribution.png")
    
    ## time_plot
    plt.figure(figsize=(10, 6))
    plt.plot(conn_times, label="Connectivity Time")
    plt.plot(metis_times, label="METIS Partition Time")
    plt.plot(louvain_times, label="Louvain Partition Time")
    plt.plot(ldg_times, label="LDG Partition Time")
    plt.plot(spectral_times, label="Spectral Partition Time")
    plt.plot(grid_times, label="Grid Partition Time")
    plt.xlabel("Structure Index")
    plt.ylabel("Time (s)")
    plt.title("Partitioning Time for Different Methods")
    plt.legend()
    plt.savefig(f"{figures_dir}/partitioning_time.png")
    
    ## bfs_time_plot
    plt.figure(figsize=(10, 6))
    plt.plot(metis_bfs_times, label="METIS BFS Time")
    plt.plot(louvain_bfs_times, label="Louvain BFS Time")
    plt.plot(ldg_bfs_times, label="LDG BFS Time")
    plt.plot(spectral_bfs_times, label="Spectral BFS Time")
    plt.plot(grid_bfs_times, label="Grid BFS Time")
    plt.xlabel("Structure Index")
    plt.ylabel("Time (s)")
    plt.title("BFS Time for Different Methods")
    plt.legend()
    plt.savefig(f"{figures_dir}/bfs_time.png")
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structs",
        type=str,
        default="./datasets/test_large.xyz",
    )
    parser.add_argument("--mp_steps", type=int, default=1)
    parser.add_argument("--granularity", type=int, default=2)
    args = parser.parse_args()
    args_dict = vars(args)
    
    main(args_dict)