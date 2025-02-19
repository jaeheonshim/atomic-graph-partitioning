import os
from tqdm import tqdm
import numpy as np
import torch
from partitioner import part_graph_extended
import networkx as nx
from ase.io import read
from ase.build import make_supercell
from mattersim.datasets.utils.convertor import GraphConvertor
from torch_geometric.utils import to_networkx
from typing import Dict
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from torch_runstats.scatter import scatter
from mattersim.utils.download_utils import download_checkpoint
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
import csv
import time

class M3GnetModified(M3Gnet):
    def forward(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].float()
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        # -------------------------------------------------------------#
        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        index_bias = torch.repeat_interleave(  # noqa: F501
            cumsum, num_three_body, dim=0
        ).unsqueeze(-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = torch.repeat_interleave(repeats=num_atoms)
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop
        for idx, conv in enumerate(self.graph_conv):
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        return atom_attr  # [batch_size]
     
def load_modified_from_checkpoint(
    load_path: str = None,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    checkpoint_folder = os.path.expanduser("~/.local/mattersim/pretrained_models")
    os.makedirs(checkpoint_folder, exist_ok=True)
    if (
        load_path is None
        or load_path.lower() == "mattersim-v1.0.0-1m.pth"
        or load_path.lower() == "mattersim-v1.0.0-1m"
    ):
        load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-1M.pth")
        if not os.path.exists(load_path):
            print(
                "The pre-trained model is not found locally, "
                "attempting to download it from the server."
            )
            download_checkpoint(
                "mattersim-v1.0.0-1M.pth", save_folder=checkpoint_folder
            )
        print(f"Loading the pre-trained {os.path.basename(load_path)} model")
    elif (
        load_path.lower() == "mattersim-v1.0.0-5m.pth"
        or load_path.lower() == "mattersim-v1.0.0-5m"
    ):
        load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-5M.pth")
        if not os.path.exists(load_path):
            print(
                "The pre-trained model is not found locally, "
                "attempting to download it from the server."
            )
            download_checkpoint(
                "mattersim-v1.0.0-5M.pth", save_folder=checkpoint_folder
            )
        print(f"Loading the pre-trained {os.path.basename(load_path)} model")
    else:
        print("Loading the model from %s" % load_path)
    assert os.path.exists(load_path), f"Model file {load_path} not found"

    checkpoint = torch.load(load_path, map_location=device)

    model = M3GnetModified(device=device, **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)

    model.eval()

    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_modified_from_checkpoint(device=device)

base_atoms = read("datasets/test.xyz")

def test_partitioning_supercell(supercell_scaling, desired_partitions = 20, neighborhood_distance = 5):
    ## Loading Data
    print("Loading Data")
    atoms = make_supercell(base_atoms, supercell_scaling)

    converter = GraphConvertor("m3gnet", 5.0, True, 4.0)

    atom_graph = converter.convert(atoms.copy(), None, None, None)

    G = to_networkx(atom_graph)

    print("Partitioning Atoms")
    partitions, extended_partitions = part_graph_extended(G, desired_partitions, neighborhood_distance)

    num_partitions = len(partitions)

    from ase import Atoms

    partitioned_atoms = []
    indices_map = [] # Table mapping each atom in each partition back to its index in the original atoms object

    for part in extended_partitions:
        current_partition = []
        current_indices_map = []
        for atom_index in part:
            current_partition.append(atoms[atom_index])
            current_indices_map.append(atoms[atom_index].index)

        atoms = Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos_center = np.mean(scaled_pos, axis=0)
        scaled_pos_offset = scaled_pos_center - 0.5
        scaled_pos = np.mod(scaled_pos - scaled_pos_offset, 1)
        atoms.set_scaled_positions(scaled_pos)
        
        partitioned_atoms.append(atoms)
        indices_map.append(current_indices_map)

    ## Inference

    from mattersim.forcefield.potential import batch_to_dict
    from torch_geometric.loader import DataLoader

    aggregated_atomic_numbers = torch.zeros((len(atoms), 1), dtype=torch.float32, device=device)
    aggregated_features = torch.zeros((len(atoms), 128), dtype=torch.float32, device=device)

    dataloader = DataLoader([converter.convert(part.copy(), None, None, None) for part in partitioned_atoms])

    partition_inference_times = []

    for part_idx, input_graph in tqdm(enumerate(dataloader), total=num_partitions):
        t0 = time.time()
        input_graph = input_graph.to(device)
        input_dict = batch_to_dict(input_graph)
        atomic_numbers = input_dict["atom_attr"]

        with torch.no_grad():
            feat = model.forward(input_dict)

        part = partitioned_atoms[part_idx]
        for j, node in enumerate(part):
            original_index = indices_map[part_idx][j]
            if original_index in partitions[part_idx]: # If the node is a root node of the partition
                aggregated_features[original_index] = feat[j]
                aggregated_atomic_numbers[original_index] = atomic_numbers[j]

        t1 = time.time()

        partition_inference_times.append(t1 - t0)

        del input_graph, input_dict, atomic_numbers, feat
        torch.cuda.empty_cache()

    atomic_numbers = aggregated_atomic_numbers.squeeze(1).long()
    batch = torch.zeros((len(atoms)), dtype=torch.int64, device=device)

    energy = model.final(aggregated_features).view(-1)
    energy = model.normalizer(energy, atomic_numbers)
    energy = scatter(energy, batch, dim=0, dim_size=1)

    ## Benchmark Inference
    torch.cuda.empty_cache()

    t0 = time.time()
    potential = Potential.from_checkpoint(device=device)
    dataloader = build_dataloader([atoms], only_inference=True)

    with torch.no_grad():
        predictions = potential.predict_properties(dataloader, include_forces=False, include_stresses=False)
        benchmark_energy = predictions[0][0]

    energy_error_abs = torch.abs(benchmark_energy - energy).item()
    energy_error_pct = torch.abs((benchmark_energy - energy) / benchmark_energy).item() * 100
    t1 = time.time()

    benchmark_time = t1 - t0

    return {
        "num_atoms": len(atoms),
        "avg_partition_size": sum(len(x) for x in extended_partitions) / num_partitions,
        "partition_energy": energy.item(),
        "benchmark_energy": benchmark_energy,
        "energy_error_abs": energy_error_abs,
        "energy_error_pct": energy_error_pct,
        "avg_partition_time": sum(partition_inference_times) / len(partition_inference_times),
        "benchmark_time": benchmark_time
    }

results = []
for x in range(1, 7):
    for yz in range(x, x + 2):
        for i in range(3):
            results.append(test_partitioning_supercell([[x, 0, 0], [0, yz, 0], [0, 0, yz]]))

with open('mattersim_test_results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, results[0].keys())
    writer.writeheader()
    writer.writerows(results)