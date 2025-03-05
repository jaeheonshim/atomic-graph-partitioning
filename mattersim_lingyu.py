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

from partitioner import part_graph_extended
import networkx as nx


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mattersim-1m")
parser.add_argument("--desired_partitions", type=int, default=20)
parser.add_argument("--num_message_passing", type=int, default=3)
parser.add_argument("--expected_unit_size", type=int, default=5000)
parser.add_argument("--num_samples", type=int, default=20)
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

supercell_sizes = []
parts_avg_sizes = []
ori_energies = []
ori_forces = []
part_energies = []
part_forces = []
ori_infer_times = []
part_process_times = []
part_infer_times = []
for atoms in tqdm(atoms_list):
    max_supercell_size = int(np.ceil(np.cbrt(expected_unit_size / len(atoms))))
    min_supercell_size = int(np.ceil(np.cbrt(100 / len(atoms))))
    for supercell_size in range(min_supercell_size, max_supercell_size):
        try:
            ## ===================== Expand Supercell And Partition =====================
            atoms = make_supercell(atoms, [[int(supercell_size), 0, 0], [0, int(supercell_size), 0], [0, 0, int(supercell_size)]])
            converter = GraphConvertor("m3gnet", 5.0, True, 4.0)
            
            ## ===================== Perform Original Inferencing =====================
            time_start = time.time()
            dataloader = DataLoader([converter.convert(atoms.copy(), None, None, None)], batch_size=1)
            energy_ori = 0
            forces_ori = np.zeros((len(atoms), 3))
            for input_graph in dataloader:
                input_graph = input_graph.to(device)
                input_dict = batch_to_dict(input_graph)
                output = potential(input_dict, include_forces=True, include_stresses=False)
                energy_ori = output["energies"].detach().cpu().numpy()
                forces_ori = output["forces"].detach().cpu().numpy()
                energy_ori = energy_ori.item()
                forces_ori = forces_ori.reshape(-1, 3)
                torch.cuda.empty_cache()
            time_end = time.time()
            ori_infer_time = time_end - time_start
            
            ## ===================== Perform Partitioned Inferencing =====================
            time_start = time.time()
            length = len(atoms)
            atom_graph = converter.convert(atoms.copy(), None, None, None)
            G = to_networkx(atom_graph)

            partitions, extended_partitions = part_graph_extended(G, desired_partitions, num_message_passing)
            num_partitions = len(partitions)

            partitioned_atoms = []
            indices_map = [] # Table mapping each atom in each partition back to its index in the original atoms object
            root_node_indices_list = []
            for part, extended_part in zip(partitions, extended_partitions):
                current_partition = []
                current_indices_map = []
                root_node_indices = []
                for i, atom_index in enumerate(extended_part):
                    current_partition.append(atoms[atom_index])
                    current_indices_map.append(atoms[atom_index].index) # type: ignore
                    if atom_index in part:
                        root_node_indices.append(i)
                partitioned_atoms.append(Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)) # It's important to pass atoms.cell and atoms.pbc here
                indices_map.append(current_indices_map)
                root_node_indices_list.append(root_node_indices)
            time_end = time.time()
            part_process_time = time_end - time_start
            
            time_start = time.time()
            dataloader = DataLoader([converter.convert(part.copy(), None, None, None) for part in partitioned_atoms], batch_size=1)
            energies_parts = np.zeros(len(atoms))
            forces_parts = np.zeros((len(atoms), 3))
            for part_idx, input_graph in enumerate(dataloader):
                input_graph = input_graph.to(device)
                input_dict = batch_to_dict(input_graph)
                output = potential(
                    input_dict, 
                    include_forces=True, 
                    include_stresses=False,
                    root_node_indices=torch.tensor(root_node_indices_list[part_idx])
                )
                energies_i = output["energies_i"].detach().cpu().numpy()
                forces = output["forces"].detach().cpu().numpy()
                part = partitioned_atoms[part_idx]
                for j in range(len(part)):
                    original_index = indices_map[part_idx][j]
                    if original_index in partitions[part_idx]:
                        energies_parts[original_index] = energies_i[j]
                    forces_parts[original_index] += forces[j]
                torch.cuda.empty_cache()
            energy_part = np.sum(energies_parts).item()
            time_end = time.time()
            part_infer_time = time_end - time_start
            
            ## ===================== Record =====================
            supercell_sizes.append(len(atoms))
            parts_avg_sizes.append(sum([len(partition) for partition in partitions]) / num_partitions)
            ori_energies.append(energy_ori)
            ori_forces.append(forces_ori)
            part_energies.append(energy_part)
            part_forces.append(forces_parts)
            ori_infer_times.append(ori_infer_time)
            part_process_times.append(part_process_time)
            part_infer_times.append(part_infer_time)
        except Exception as e:
            print(f"error for {supercell_size}")
            ## check error is due to memory or not
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory")
                break
            else:
                print(e)
                break
    
# ================================ Evaluation ================================
# ori_energies = np.array(ori_energies)
# ori_forces = np.vstack(ori_forces)
# part_energies = np.array(part_energies)
# part_forces = np.vstack(part_forces)

# e_mae = torch.nn.L1Loss()(torch.tensor(ori_energies), torch.tensor(part_energies)).item()
# f_mae = torch.nn.L1Loss()(torch.tensor(ori_forces), torch.tensor(part_forces)).item()

e_maes = []
f_maes = []
e_mae_ratios = []
f_mae_ratios = []
print(len(supercell_sizes), len(ori_energies), len(ori_forces), len(part_energies), len(part_forces))
for i in range(len(supercell_sizes)):
    ori_energy = ori_energies[i]
    ori_force = ori_forces[i]
    part_energy = part_energies[i]
    part_force = part_forces[i]
    num_atoms = supercell_sizes[i]
    e_mae = np.abs(ori_energy - part_energy)/num_atoms
    f_mae = np.mean(np.linalg.norm(ori_force - part_force, ord=1, axis=1))
    e_maes.append(e_mae)
    f_maes.append(f_mae)
    e_mae_ratio = np.abs(ori_energy - part_energy) / np.abs(ori_energy)
    f_mae_ratio = np.mean(np.linalg.norm(ori_force - part_force, ord=1, axis=1) / np.linalg.norm(ori_force, ord=1, axis=1))
    e_mae_ratios.append(e_mae_ratio)
    f_mae_ratios.append(f_mae_ratio)


supercell_sizes = np.array(supercell_sizes)
parts_avg_sizes = np.array(parts_avg_sizes)
e_maes = np.array(e_maes)
f_maes = np.array(f_maes)
e_mae_ratios = np.array(e_mae_ratios)
f_mae_ratios = np.array(f_mae_ratios)
ori_infer_times = np.array(ori_infer_times)
part_process_times = np.array(part_process_times)
part_infer_times = np.array(part_infer_times)
sorted_indices = np.argsort(supercell_sizes)
supercell_sizes = supercell_sizes[sorted_indices]
e_maes = e_maes[sorted_indices]
f_maes = f_maes[sorted_indices]
e_mae_ratios = e_mae_ratios[sorted_indices]
f_mae_ratios = f_mae_ratios[sorted_indices]
ori_infer_times = ori_infer_times[sorted_indices]
part_process_times = part_process_times[sorted_indices]
part_infer_times = part_infer_times[sorted_indices]


# ================================ Plotting ================================
os.makedirs("figures", exist_ok=True)

# Plot supercell size vs. ...
plt.figure(figsize=(18, 18))
plt.subplot(2, 2, 1)
plt.plot(supercell_sizes, e_maes, "o")
plt.xlabel("Supercell Size")
plt.ylabel("Energy MAE (eV/atom)")
plt.yscale("log")
plt.title(f"E_MAE vs Size, Avg: {np.mean(e_maes):.2e}")
plt.subplot(2, 2, 2)
plt.plot(supercell_sizes, f_maes, "o")
plt.xlabel("Supercell Size")
plt.ylabel("Force MAE (eV/Angstrom)")
plt.yscale("log")
plt.title(f"F_MAE vs Size, Avg: {np.mean(f_maes):.2e}")
plt.subplot(2, 2, 3)
plt.plot(supercell_sizes, e_mae_ratios, "o")
plt.xlabel("Supercell Size")
plt.ylabel("Energy MAE Ratio")
plt.yscale("log")
plt.title(f"E_MAE_Ratio vs Size, Avg: {np.mean(e_mae_ratios):.2e}")
plt.subplot(2, 2, 4)
plt.plot(supercell_sizes, f_mae_ratios, "o")
plt.xlabel("Supercell Size")
plt.ylabel("Force MAE Ratio")
plt.yscale("log")
plt.title(f"F_MAE_Ratio vs Size, Avg: {np.mean(f_mae_ratios):.2e}")
plt.savefig(f"figures/{MODEL}-x{desired_partitions}-{num_message_passing}MP-ErrorVsSize.png")
plt.close()

# Plot Time
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(supercell_sizes, ori_infer_times, "o", label="Original")
plt.plot(supercell_sizes, part_process_times+part_infer_times, "o", label="Partition")
plt.xlabel("Supercell Size")
plt.ylabel("Time (s)")
plt.legend()
plt.title(f"Time vs Size")
plt.subplot(1, 3, 2)
plt.plot(supercell_sizes, part_process_times, "o", label="Partition Process")
plt.plot(supercell_sizes, part_infer_times, "o", label="Partition Infer")
plt.xlabel("Supercell Size")
plt.ylabel("Time (s)")
plt.legend()
plt.title(f"Partition Time vs Size")
plt.subplot(1, 3, 3)
sort_indices = np.argsort(parts_avg_sizes)
plt.plot(parts_avg_sizes[sort_indices], part_process_times[sort_indices], "o", label="Partition Process")
plt.plot(parts_avg_sizes[sort_indices], part_infer_times[sort_indices], "o", label="Partition Infer")
plt.xlabel("Partition Size")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig(f"figures/{MODEL}-x{desired_partitions}-{num_message_passing}MP-TimeVsSize.png")
plt.close()