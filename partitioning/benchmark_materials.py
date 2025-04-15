import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from ase import Atoms
from ase.visualize import view
from ase.build import make_supercell
from ase.io import read

from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os
from mp_api.client import MPRester
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

import csv
import argparse

from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from mattersim.datasets.utils.convertor import GraphConvertor

from partitioning import part_spectral, part_metis, part_grid, part_metis_unweighted

load_dotenv()

API_KEY=os.getenv("MP_API_KEY")

def get_atoms(id):
    with MPRester(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(id)
        
        atoms = AseAtomsAdaptor.get_atoms(structure)
        
        return atoms

mattersim_converter = GraphConvertor("m3gnet", 5.0, True, 4.0)

def atoms_to_graph_orb(atoms):
    graph = ase_atoms_to_atom_graphs(atoms)
    senders = graph.senders
    receivers = graph.receivers
    edge_feats = graph.edge_features

    G = nx.Graph()
    G.add_nodes_from(range(graph.n_node))
    G.graph['edge_weight_attr'] = 'weight'
    
    min_dist = min(edge_feats['r'])

    for i, u in enumerate(senders):
        G.add_edge(u.item(), receivers[i].item(), weight=int(edge_feats['r'][i] / min_dist * 1000))

    return G 

def atoms_to_graph_mattersim(atoms):
    return to_networkx(mattersim_converter.convert(atoms, None, None, None))

def total_node_count(G, partitions, extended_partitions):
    """
    Returns the total number of nodes including overlapping nodes in extended
    partitions
    """
    
    return sum(len(x) for x in extended_partitions)

def root_node_count(G, partitions, extended_partitions):
    """
    Returns the total number of root nodes (the number of vertices in the
    original graph)
    """
    
    return sum(len(x) for x in partitions)

def extended_ratio(G, partitions, extended_partitions):
    """
    Returns the ratio of extended nodes to the total number of nodes (higher 
    ratio means more neighbors were captured in the partition, which we don't
    want)
    
    i.e. how many of our nodes are 'redundant'?
    """
    
    root = root_node_count(G, partitions, extended_partitions)
    total = total_node_count(G, partitions, extended_partitions)
    
    return (total - root) / (total)

def core_partition_stats(G, partitions, extended_partitions):
    """
    Returns statistics about the extended partitions (max, min, mean, std, 
    range)
    """
    
    partition_sizes = [len(p) for p in partitions]
    sizes = np.array(partition_sizes)
    
    return sizes.max(), sizes.min(), sizes.mean(), sizes.std(), (sizes.max() - sizes.mean())

def extended_partition_stats(G, partitions, extended_partitions):
    """
    Returns statistics about the extended partitions (max, min, mean, std, 
    range)
    """
    
    partition_sizes = [len(p) for p in extended_partitions]
    sizes = np.array(partition_sizes)
    
    return sizes.max(), sizes.min(), sizes.mean(), sizes.std(), (sizes.max() - sizes.mean())

def num_cut_edges(G, partitions, extended_partitions):
    """
    Returns the number of edges that were cut between partitions
    """
    
    node_to_part = {}
    for i, part in enumerate(partitions):
        for node in part:
            node_to_part[node] = i

    cut_edges = set()

    # Check each edge to see if it crosses partitions
    for u, v in G.edges():
        pu = node_to_part.get(u)
        pv = node_to_part.get(v)
        if pu is not None and pv is not None and pu != pv:
            cut_edges.add(tuple(sorted((u, v))))

    return len(cut_edges)

def sum_cut_edge_weights(G, partitions, extended_partitions):
    """
    For all cut edges, return sum(x) where x is the weight of each edge
    """
    
    cut_edges = set()
    node_to_partition = {node: i for i, part in enumerate(partitions) for node in part}
    for u, v in G.edges():
        if node_to_partition.get(u) != node_to_partition.get(v):
            cut_edges.add(tuple(sorted((u, v))))
    return sum(G.edges[u, v].get('weight', 0.0) for u, v in cut_edges)

def sum_inverse_cut_edge_weights(G, partitions, extended_partitions):
    """
    For all cut edges, return sum(1/x) where x is the weight of each edge
    
    Maybe minimizing this quantity will help because that means we are
    cutting fewer edges of atoms that are closer together (and thus
    more likely to have a greater impact on each other)
    """
    
    cut_edges = set()

    node_to_partition = {}
    for i, part in enumerate(partitions):
        for node in part:
            node_to_partition[node] = i

    for u, v in G.edges():
        p_u = node_to_partition.get(u)
        p_v = node_to_partition.get(v)
        if p_u is not None and p_v is not None and p_u != p_v:
            cut_edges.add(tuple(sorted((u, v))))

    # Sum inverse of edge weights for cut edges
    total_inverse_weight = 0.0
    for u, v in cut_edges:
        w = G.edges[u, v].get('weight', None)
        if w is not None and w > 0:
            total_inverse_weight += 1.0 / (w)

    return total_inverse_weight

def benchmark_trial(atoms, atoms_to_graph_fn, distance=4, granularity=2):
    G = atoms_to_graph_fn(atoms)
    
    print(len(G.edges))

    extended_partition_results = []
    results = []

    for method_name, partition_fn in [
        ("metis", part_metis),
        ("spectral", part_spectral),
        ("grid", part_grid),
        ("metis_unweighted", part_metis_unweighted)
    ]:
        if method_name == 'spectral' and len(G.nodes) > 20000:
            continue
        
        try:
            partitions, extended_partitions = partition_fn(atoms, G, granularity ** 3, distance=distance)

            # Core stats
            core_max, core_min, core_mean, core_std, core_range = core_partition_stats(G, partitions, extended_partitions)
            # Extended stats
            ext_max, ext_min, ext_mean, ext_std, ext_range = extended_partition_stats(G, partitions, extended_partitions)

            result = {
                'method': method_name,
                'node_count': total_node_count(G, partitions, extended_partitions),
                'root_node_count': root_node_count(G, partitions, extended_partitions),
                'extended_ratio': extended_ratio(G, partitions, extended_partitions),
                'cut_edges': num_cut_edges(G, partitions, extended_partitions),
                'cut_weight_sum': sum_cut_edge_weights(G, partitions, extended_partitions),
                'inverse_cut_weight_sum': sum_inverse_cut_edge_weights(G, partitions, extended_partitions),

                # Core stats
                'core_max': core_max,
                'core_min': core_min,
                'core_mean': core_mean,
                'core_std': core_std,
                'core_range': core_range,

                # Extended stats
                'ext_max': ext_max,
                'ext_min': ext_min,
                'ext_mean': ext_mean,
                'ext_std': ext_std,
                'ext_range': ext_range,
            }

            results.append(result)

            for part, ext_part in zip(partitions, extended_partitions):
                extended_partition_results.append({
                    'method': method_name,
                    'core_n': len(part),
                    'n': len(ext_part)
                })

        except Exception as e:
            print(f"[{method_name}] Partitioning failed: {e}")

    return results, extended_partition_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_index", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--material_list", type=str, required=True)
    parser.add_argument("--num_atoms", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    material_ids = []

    with open(args.material_list, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            material_ids.append(row[0])
            
    start = args.chunk_index * args.chunk_size
    end = start + args.chunk_size
            
    material_ids = np.array(material_ids[start:end])

    all_results_orb = []
    all_partition_results_orb = []
    all_results_mattersim = []
    all_partition_results_mattersim = []

    for id in material_ids:
        atoms = get_atoms(id)
        scale = int((args.num_atoms / len(atoms)) ** (1/3)) + 1
        atoms = make_supercell(atoms, ((scale, 0, 0), (0, scale, 0), (0, 0, scale)))
        
        result_orb = benchmark_trial(atoms, atoms_to_graph_orb)
        all_results_orb.extend(result_orb[0])
        all_partition_results_orb.extend(result_orb[1])
        
        result_mattersim = benchmark_trial(atoms, atoms_to_graph_mattersim)
        all_results_mattersim.extend(result_mattersim[0])
        all_partition_results_mattersim.extend(result_mattersim[1])

    all_df_orb = pd.DataFrame(all_results_orb)
    partition_df_orb = pd.DataFrame(all_partition_results_orb)
    
    all_df_mattersim = pd.DataFrame(all_results_mattersim)
    partition_df_mattersim = pd.DataFrame(all_partition_results_mattersim)
    
    all_df_orb.to_csv(f"{args.output}_n_{args.num_atoms}_full_orb.csv", index=False)
    partition_df_orb.to_csv(f"{args.output}_n_{args.num_atoms}_partitions_orb.csv", index=False)
    all_df_mattersim.to_csv(f"{args.output}_n_{args.num_atoms}_full_mattersim.csv", index=False)
    partition_df_mattersim.to_csv(f"{args.output}_n_{args.num_atoms}_partitions_mattersim.csv", index=False)
    
if __name__ == "__main__":
    main()