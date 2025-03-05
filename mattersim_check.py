import os
import time

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from mattersim.datasets.utils.convertor import GraphConvertor, get_fixed_radius_bonding
from torch_geometric.utils import to_networkx
from models.mattersim_potential import PartitionPotential
from mattersim.forcefield.potential import batch_to_dict
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
import argparse

from tqdm import tqdm

from partitioner import part_graph_extended, part_completeness_check
import networkx as nx


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mattersim-1m")
parser.add_argument("--desired_partitions", type=int, default=20)
parser.add_argument("--num_message_passing", type=int, default=3)
parser.add_argument("--expected_unit_size", type=int, default=30000)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda:2")
args = parser.parse_args()

MODEL=args.model
desired_partitions = args.desired_partitions
num_message_passing = args.num_message_passing
expected_unit_size = args.expected_unit_size
device = args.device
num_samples = args.num_samples
potential = PartitionPotential.from_checkpoint(load_training_state=False) # type: ignore
potential = potential.to(device)
atoms_list: list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/mptraj_val.xyz", index=":") # type: ignore
random_indices = np.random.choice(len(atoms_list), num_samples, replace=False)
atoms_list = [atoms_list[i] for i in random_indices]

converter = GraphConvertor(
    model_type="m3gnet",
    twobody_cutoff=potential.model.model_args["cutoff"],
    has_threebody=True,
    threebody_cutoff=potential.model.model_args["threebody_cutoff"],
)


for atoms in tqdm(atoms_list):
    max_supercell_size = int(np.ceil(np.cbrt(expected_unit_size / len(atoms))))
    min_supercell_size = int(np.ceil(np.cbrt(1000 / len(atoms))))
    for supercell_size in range(min_supercell_size, max_supercell_size):
        atoms = make_supercell(atoms, [[int(supercell_size), 0, 0], [0, int(supercell_size), 0], [0, 0, int(supercell_size)]])
        if len(atoms) > expected_unit_size:
            break
        atom_graph = converter.convert(atoms.copy(), None, None, None)
        G = to_networkx(atom_graph)
        edge_index = atom_graph.edge_index.numpy() # type: ignore
        partitions, extended_partitions = part_graph_extended(G, desired_partitions, num_message_passing)
        check_result = part_completeness_check(
            len(atoms), 
            num_message_passing, 
            partitions,
            extended_partitions,
            edge_index,
        )
        assert check_result, "Partition completeness check failed"
        print("Partition completeness check passed")
        partitioned_atoms = []
        indices_map = [] # Table mapping each atom in each partition back to its index in the original atoms object
        for part in extended_partitions:
            current_partition = []
            current_indices_map = []
            for atom_index in part:
                current_partition.append(atoms[atom_index])
                current_indices_map.append(atoms[atom_index].index) # type: ignore
            partitioned_atoms.append(Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)) # It's important to pass atoms.cell and atoms.pbc here
            indices_map.append(current_indices_map)
        
        atom_attrs_ori = None
        atom_attrs_part = None
        try:
            dataloader = DataLoader([converter.convert(atoms.copy(), None, None, None)], batch_size=1)
            for input_graph in dataloader:
                input_graph = input_graph.to(device)
                input_dict = batch_to_dict(input_graph)
                _, internal_attrs = potential(input_dict, include_forces=True, include_stresses=False, return_intermediate=True)
                atom_attrs_ori = internal_attrs["node_attr_2"].detach().cpu().numpy()
            
            atom_attrs_part = np.zeros_like(atom_attrs_ori)
            dataloader = DataLoader([converter.convert(part.copy(), None, None, None) for part in partitioned_atoms], batch_size=1)
            for part_idx, input_graph in enumerate(dataloader):
                input_graph = input_graph.to(device)
                input_dict = batch_to_dict(input_graph)
                _, internal_attrs = potential(input_dict, include_forces=True, include_stresses=False, return_intermediate=True)
                atom_attrs = internal_attrs["node_attr_2"].detach().cpu().numpy()
                part = partitioned_atoms[part_idx]
                for j in range(len(part)):
                    original_index = indices_map[part_idx][j]
                    if original_index in partitions[part_idx]:
                        atom_attrs_part[original_index] = atom_attrs[j]
        except Exception as e:
            print(f"error for {supercell_size}")
            ## check error is due to memory or not
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory")
                break
            else:
                print(e)
                break
        
        if atom_attrs_ori is not None and atom_attrs_part is not None:
            assert np.allclose(atom_attrs_ori, atom_attrs_part, atol=1e-5), "Partitioned atom attributes do not match original atom attributes"
            print("Partitioned atom attributes match original atom attributes")