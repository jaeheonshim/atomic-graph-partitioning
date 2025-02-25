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
        desired_partitions = 20
        neighborhood_distance = 4
        partitions, extended_partitions = part_graph_extended(G, desired_partitions, neighborhood_distance)

        num_partitions = len(partitions)

        partitioned_atoms = []
        indices_map = [] # Table mapping each atom in each partition back to its index in the original atoms object

        for part in extended_partitions:
            current_partition = []
            current_indices_map = []
            for atom_index in part:
                current_partition.append(atoms[atom_index])
                current_indices_map.append(atoms[atom_index].index)

            partitioned_atoms.append(ase.Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)) # It's important to pass atoms.cell and atoms.pbc here
            indices_map.append(current_indices_map)

        ### Graph Regressor
        all_embeddings = torch.zeros((len(atoms), self.model_adapter.embedding_size), dtype=torch.float32, device=self.model_adapter.device)

        for i, part in tqdm(enumerate(partitioned_atoms), total=num_partitions):
            input_graph = self.model_adapter.atoms_to_graph(part)

            part_embeddings = self.model_adapter.forward_graph(input_graph)

            for j, node in enumerate(part):
                original_index = indices_map[i][j]
                if original_index in partitions[i]: # If the node is a root node of the partition
                    all_embeddings[original_index] = part_embeddings[j]

        ### Extract Energy
        energy = self.model_adapter.forward_energy(all_embeddings, atoms)

        return {
            "energy": energy
        }
    
from ase.io import read
from ase.build import make_supercell
from orb_models.forcefield.base import AtomGraphs
from implementations.orb import OrbModelAdapter

orb_adapter = OrbModelAdapter()
inference = AtomicPartitionInference(orb_adapter)

atoms = read("datasets/test.xyz")
atoms = make_supercell(atoms, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(inference.run(atoms))