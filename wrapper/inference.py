from wrapper.adapter import AtomicModelAdapter
from wrapper.partitioner import part_graph_extended

import time
import gc

import ase
import numpy as np
import torch
from tqdm import tqdm

class AtomicPartitionInference:
    def __init__(self, model_adapter: AtomicModelAdapter):
        self.model_adapter = model_adapter

    def run(self, 
            atoms: ase.Atoms,
            *,
            desired_partitions: int,
            parts_per_batch: int = 1
        ):

        print()
        print("=" * 45)
        print("Beginning partitioned inference")
        print(f"{'- Number of atoms':<33}: {len(atoms):>10}")
        print(f"{'- Desired number of partitions':<33}: {desired_partitions:>10}")
        print(f"{'- Number of partitions per batch':<33}: {parts_per_batch:>10}")
        print("=" * 45)
        print()

        ### Data Preparation
        graph = self.model_adapter.atoms_to_graph(atoms)
        G = self.model_adapter.graph_to_networkx(graph)

        ### Partitioning
        print("Partitioning graph...")
        partition_set, extended_partition_set = part_graph_extended(G, desired_partitions, self.model_adapter.num_message_passing)
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

        self.model_adapter.init_partition(atoms, indices_map, partition_roots)

        print(f"Partitioning complete! Created {num_partitions} partitions")

        ### Graph Regressor
        print("Starting inference...")
        times = []
        all_embeddings = torch.zeros((len(atoms), self.model_adapter.embedding_size), dtype=torch.float32, device=self.model_adapter.device)

        for i in tqdm(range(0, len(partitioned_atoms), parts_per_batch)):
            start = time.time()

            parts = partitioned_atoms[i:i+parts_per_batch]
            input_graph = [self.model_adapter.atoms_to_graph(part) for part in parts]

            part_embeddings = self.model_adapter.forward_graph(input_graph, list(range(i, i + len(input_graph))))

            for j in range(0, len(part_embeddings)):
                reverse_indices = indices_map[i+j]
                for k in range(0, len(part_embeddings[j])):
                    if partition_roots[i+j][k]:
                        all_embeddings[reverse_indices[k]] = part_embeddings[j][k]

            end = time.time()

            del part_embeddings, input_graph
            gc.collect()
            torch.cuda.empty_cache()

            times.append(end - start)
        print("Inference complete!")

        ### Extract Energy
        energy = self.model_adapter.predict_energy(all_embeddings, atoms).detach().cpu().numpy()
        forces = self.model_adapter.predict_forces(all_embeddings, atoms).detach().cpu().numpy()

        return {
            "energy": energy,
            "forces": forces,
            "times": np.array(times)
        }