from wrapper.inference import AtomicPartitionInference
from wrapper.implementations.mattersim import MatterSimModelAdapter
from wrapper.implementations.orb import OrbModelAdapter

from ase.io import read
from ase.build import make_supercell

from mattersim.forcefield import MatterSimCalculator

from orb_models.forcefield import atomic_system, pretrained

import torch
import numpy as np

import csv
import time

device = 'cpu'

ATOMS_FILE = "datasets/H2O.xyz"
MAX_SUPERCELL_DIM = 2
NUM_PARTITIONS = 10

MATTERSIM_ITERATIONS = 5 # Mattersim is a little weird so I will run multiple times and average

orbff = pretrained.orb_v2(device=device)

orb_partition_inference = AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=3))
mattersim_partition_inference = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=3))

mp_list = [4]
fields = ['num_atoms', 'num_parts', 'num_mp', 'energy_error_abs', 'energy_error_pct', 'forces_error_max', 'forces_error_mae', 'forces_error_mape', 'forces_error_ratio','forces_error_mse', 'forces_error_rms', 'benchmark_time', 'all_partition_time', 'avg_partition_time']
orb_rows = []
mattersim_rows = []

def get_mattersim_benchmark(atoms):
    mattersim_calc = MatterSimCalculator(compute_stress=False)
    atoms.calc = mattersim_calc

    return {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces()
    }

def get_orb_benchmark(atoms):
    input_graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)
    result = orbff.predict(input_graph)

    return {
        "energy": result["graph_pred"],
        "forces": result["node_pred"]
    }

def run_orb_error_test(supercell_scaling):
    atoms = read(ATOMS_FILE)
    atoms = make_supercell(atoms, supercell_scaling)

    benchmark = get_orb_benchmark(atoms)

    for mp in mp_list:
        orb_partition_inference.model_adapter.num_message_passing = mp
        result = orb_partition_inference.run(atoms, desired_partitions=NUM_PARTITIONS)
        
        orb_rows.append([
            len(atoms),
            mp,
            abs(benchmark["energy"] - result["energy"]).item(),
            abs((benchmark["energy"] - result["energy"]) / benchmark["energy"]).item() * 100,
            torch.max(torch.abs(benchmark["forces"] - result["forces"])).item(),
            torch.mean(torch.abs(benchmark["forces"] - result["forces"])).item(),
            torch.mean(torch.abs((benchmark["forces"] - result["forces"]) / benchmark["forces"])).item() * 100,
            torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2)).item(),
            torch.sqrt(torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2))).item(),
        ])

def run_mattersim_error_test(atoms, num_parts, num_mp):
    benchmark_energy = []
    benchmark_forces = []

    start = time.time()
    for _ in range(MATTERSIM_ITERATIONS):
        benchmark = get_mattersim_benchmark(atoms)
        benchmark_energy.append(benchmark["energy"])
        benchmark_forces.append(benchmark["forces"])
    end = time.time()

    avg_benchmark_time = (end - start) / MATTERSIM_ITERATIONS

    benchmark_energy = np.mean(benchmark_energy)
    benchmark_forces = np.mean(benchmark_forces, axis=0)

    mattersim_partition_inference.model_adapter.num_message_passing = num_mp

    result_energy = []
    result_forces = []
    result_times = []
    for _ in range(MATTERSIM_ITERATIONS):
        result = mattersim_partition_inference.run(atoms, desired_partitions=NUM_PARTITIONS)
        result_energy.append(result["energy"])
        result_forces.append(result["forces"])
        result_times.append(result["times"])

    result_energy = np.mean(result_energy)
    result_forces = np.mean(result_forces, axis=0)
    result_times = np.mean(result_times, axis=0)
    
    mattersim_rows.append([
        len(atoms),
        num_parts,
        num_mp,
        abs(benchmark_energy - result_energy).item(),
        abs((benchmark_energy - result_energy) / benchmark_energy).item() * 100,
        np.max(np.abs(benchmark_forces - result_forces)).item(),
        np.mean(np.abs(benchmark_forces - result_forces)).item(),
        np.mean(np.abs((benchmark_forces - result_forces) / benchmark_forces)).item() * 100,
        np.mean(np.linalg.norm(benchmark_forces - result_forces, ord=1, axis=1) / np.linalg.norm(benchmark_forces, ord=1, axis=1)),
        np.mean((benchmark_forces - result_forces) ** 2).item(),
        np.sqrt(np.mean((benchmark_forces - result_forces) ** 2)).item(),
        avg_benchmark_time,
        np.sum(result_times),
        np.mean(result_times)

    ])

def write_csv():
    with open('orb_results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(orb_rows)

    with open('mattersim_results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(mattersim_rows)
        
for x in range(1, MAX_SUPERCELL_DIM):
    for y in range(x, x + 2):
        # run_orb_error_test(((x, 0, 0), (0, y, 0), (0, 0, y)))
        run_mattersim_error_test(read(ATOMS_FILE), 20, 5)
        write_csv()