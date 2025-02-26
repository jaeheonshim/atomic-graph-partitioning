from adapter import AtomicModelAdapter
from partitioner import part_graph_extended

import ase
import torch
from tqdm import tqdm

class AtomicPartitionInference:
    def __init__(self, model_adapter: AtomicModelAdapter):
        self.model_adapter = model_adapter

    def run(self, atoms):
        ### Data Preparation
        graph = self.model_adapter.atoms_to_graph(atoms)
        G = self.model_adapter.graph_to_networkx(graph)

        ### Partitioning
        desired_partitions = 2
        neighborhood_distance = 5
        partition_set, extended_partition_set = part_graph_extended(G, desired_partitions, neighborhood_distance)

        num_partitions = len(partition_set)

        partitioned_atoms = []

        indices_map = [] # Table mapping each atom in each partition back to its index in the original atoms object
        partition_roots = [] # Roots of each partition (True if root, False if not)

        for i, part in enumerate(extended_partition_set):
            current_partition = []
            current_indices_map = []

            for atom_index in part:
                current_partition.append(atoms[atom_index])
                current_indices_map.append(atom_index)

            partitioned_atoms.append(ase.Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)) # It's important to pass atoms.cell and atoms.pbc here
            indices_map.append(current_indices_map)
            partition_roots.append([j in partition_set[i] for j in current_indices_map])

        indices_map = torch.tensor(indices_map)
        partition_roots = torch.tensor(partition_roots)

        self.model_adapter.set_partition_info(atoms, indices_map, partition_roots)

        ### Graph Regressor
        all_embeddings = torch.zeros((len(atoms), self.model_adapter.embedding_size), dtype=torch.float32, device=self.model_adapter.device)

        for i, part in tqdm(enumerate(partitioned_atoms), total=num_partitions):
            input_graph = self.model_adapter.atoms_to_graph(part)

            part_embeddings = self.model_adapter.forward_graph(input_graph, i)

            all_embeddings[indices_map[i][partition_roots[i]]] = part_embeddings[partition_roots[i]]

        ### Extract Energy
        energy = self.model_adapter.predict_energy(all_embeddings, atoms)
        forces = self.model_adapter.predict_forces(all_embeddings, atoms)

        return {
            "energy": energy,
            "forces": forces[:10]
        }
    
from ase.io import read
from ase.build import make_supercell
from orb_models.forcefield.base import AtomGraphs
from implementations.orb import OrbModelAdapter
from implementations.mattersim import MatterSimModelAdapter

orb_adapter = MatterSimModelAdapter()
inference = AtomicPartitionInference(orb_adapter)

import torch
import numpy as np
import random
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

atoms = read("datasets/H2O.xyz")
print(inference.run(atoms))
atoms.calc = MatterSimCalculator(device=device)
print(atoms.get_forces()[:10])