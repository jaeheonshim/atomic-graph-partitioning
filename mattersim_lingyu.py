import os

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from mattersim.datasets.utils.convertor import GraphConvertor
from torch_geometric.utils import to_networkx
from models.mattersim_potential import PartitionPotential
from mattersim.forcefield.potential import batch_to_dict
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt

from tqdm import tqdm

from partitioner import part_graph_extended
import networkx as nx

MODEL="mattersim-1m"
desired_partitions = 20
num_message_passing = 3
expected_unit_size = 10000
device = "cuda:1" if torch.cuda.is_available() else "cpu"
potential = PartitionPotential.from_checkpoint(load_training_state=False) # type: ignore
potential = potential.to(device)
atoms_list: list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/mptraj_val.xyz", index=":") # type: ignore
random_indices = np.random.choice(len(atoms_list), 500, replace=False)
atoms_list = [atoms_list[i] for i in random_indices]

supercell_sizes = []
parts_avg_sizes = []
ori_energies = []
ori_forces = []
part_energies = []
part_forces = []
for atoms in tqdm(atoms_list):
    try:
        ## ===================== Expand Supercell And Partition =====================
        supercell_size = np.ceil(np.cbrt(expected_unit_size / len(atoms))) - 1
        atoms = make_supercell(atoms, [[int(supercell_size), 0, 0], [0, int(supercell_size), 0], [0, 0, int(supercell_size)]])
        supercell_sizes.append(len(atoms))
        
        converter = GraphConvertor("m3gnet", 5.0, True, 4.0)
        length = len(atoms)
        atom_graph = converter.convert(atoms.copy(), None, None, None)
        G = to_networkx(atom_graph)

        partitions, extended_partitions = part_graph_extended(G, desired_partitions, num_message_passing)
        num_partitions = len(partitions)
        parts_avg_sizes.append(sum([len(partition) for partition in partitions]) / num_partitions)

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
        
        ## ===================== Perform Inferencing =====================
        dataloader = DataLoader([converter.convert(part.copy(), None, None, None) for part in partitioned_atoms], batch_size=1)
        energies_parts = np.zeros(len(atoms))
        forces_parts = np.zeros((len(atoms), 3))
        for part_idx, input_graph in enumerate(dataloader):
            input_graph = input_graph.to(device)
            input_dict = batch_to_dict(input_graph)
            output = potential(input_dict, include_forces=True, include_stresses=False)
            energies_i = output["energies_i"].detach().cpu().numpy()
            forces = output["forces"].detach().cpu().numpy()
            part = partitioned_atoms[part_idx]
            for j in range(len(part)):
                original_index = indices_map[part_idx][j]
                if original_index in partitions[part_idx]:
                    energies_parts[original_index] = energies_i[j]
                    forces_parts[original_index] = forces[j]
            torch.cuda.empty_cache()
        energy_part = np.sum(energies_parts).item()
            
        dataloader = DataLoader([converter.convert(atoms.copy(), None, None, None)], batch_size=1)
        energy = 0
        forces = np.zeros((len(atoms), 3))
        for input_graph in dataloader:
            input_graph = input_graph.to(device)
            input_dict = batch_to_dict(input_graph)
            output = potential(input_dict, include_forces=True, include_stresses=False)
            energy = output["energies"].detach().cpu().numpy()
            forces = output["forces"].detach().cpu().numpy()
            energy = energy.item()
            forces = forces.reshape(-1, 3)
            torch.cuda.empty_cache()
        ori_energies.append(energy)
        ori_forces.append(forces)
        part_energies.append(energy_part)
        part_forces.append(forces_parts)
    except Exception as e:
        print(f"Error: {e}")
        continue
    
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
for i in range(len(atoms_list)):
    num_atoms = supercell_sizes[i]
    ori_energy = ori_energies[i]
    ori_force = ori_forces[i]
    part_energy = part_energies[i]
    part_force = part_forces[i]
    e_mae = np.abs(ori_energy - part_energy)/num_atoms
    f_mae = np.mean(np.linalg.norm(ori_force - part_force, ord=1, axis=1))
    e_maes.append(e_mae)
    f_maes.append(f_mae)
    e_mae_ratio = np.abs(ori_energy - part_energy) / np.abs(ori_energy)
    f_mae_ratio = np.mean(np.linalg.norm(ori_force - part_force, ord=1, axis=1) / np.linalg.norm(ori_force, ord=1, axis=1))
    e_mae_ratios.append(e_mae_ratio)
    f_mae_ratios.append(f_mae_ratio)
supercell_sizes = np.array(supercell_sizes)
e_maes = np.array(e_maes)
f_maes = np.array(f_maes)
e_mae_ratios = np.array(e_mae_ratios)
f_mae_ratios = np.array(f_mae_ratios)


# ================================ Plotting ================================
os.makedirs("figures", exist_ok=True)

# Plot supercell size vs. ...
sorted_indices = np.argsort(supercell_sizes)
supercell_sizes = supercell_sizes[sorted_indices]
e_maes = e_maes[sorted_indices]
f_maes = f_maes[sorted_indices]
e_mae_ratios = e_mae_ratios[sorted_indices]
f_mae_ratios = f_mae_ratios[sorted_indices]
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
plt.savefig(f"figures/{MODEL}_{num_message_passing}MP_error_vs_size.png")

# # Plot corelation
# plt.subplot(2, 2, 1)
# sorted_indices = np.argsort(e_maes)
# e_maes = e_maes[sorted_indices]
# f_maes = f_maes[sorted_indices]
# e_mae_ratios = e_mae_ratios[sorted_indices]
# f_mae_ratios = f_mae_ratios[sorted_indices]