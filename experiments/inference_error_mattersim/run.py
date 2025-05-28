from main.inference import AtomicPartitionInference
from main.implementations.mattersim import MatterSimModelAdapter
from main.implementations.orb import OrbModelAdapter

from ase import Atoms
from ase.io import read
from ase.build import make_supercell

from mattersim.forcefield import MatterSimCalculator

from orb_models.forcefield import atomic_system, pretrained

import torch
import numpy as np

import json
import csv
import time
import os

import argparse

MATTERSIM_RESULTS = "results/test/mattersim_results_randomized.csv"
ORB_RESULTS = "results/test/orb_results_randomized.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

ATOMS_FILE = "datasets/test.xyz"
NUM_PARTITIONS = 60
mp_list = [2,3,4,5,6,7,8]
box_sizes = [10, 25, 50, 100]

MATTERSIM_ITERATIONS = 1 # Mattersim is a little weird so I will run multiple times and average

orbff = pretrained.orb_v2(device=device)

orb_partition_inference = AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=3))
mattersim_partition_inference = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=3))

fields = ['num_atoms', 'num_parts', 'avg_part_size', 'num_mp', 'energy_error_abs', 'energy_error_pct', 'forces_error_max', 'forces_error_mae', 'forces_error_mape', 'forces_error_ratio','forces_error_mse', 'forces_error_rms', 'benchmark_time', 'all_partition_time', 'avg_partition_time', 'box_size']
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
        "energy": result["graph_pred"].detach().cpu(),
        "forces": result["node_pred"].detach().cpu()
    }

def run_orb_error_test(atoms, num_parts, num_mp):
    start = time.time()
    benchmark = get_orb_benchmark(atoms)
    end = time.time()

    benchmark_time = end - start

    orb_partition_inference.model_adapter.num_message_passing = num_mp
    result = orb_partition_inference.run(atoms, desired_partitions=num_parts)
    
    row = [
        len(atoms),
        num_parts,
        np.mean(result['partition_sizes']),
        num_mp,
        abs(benchmark["energy"] - result["energy"]).item(),
        abs((benchmark["energy"] - result["energy"]) / benchmark["energy"]).item() * 100,
        torch.max(torch.abs(benchmark["forces"] - result["forces"])).item(),
        torch.mean(torch.abs(benchmark["forces"] - result["forces"])).item(),
        torch.mean(torch.abs((benchmark["forces"] - result["forces"]) / benchmark["forces"])).item() * 100,
        np.mean(np.linalg.norm(benchmark["forces"] - result["forces"], ord=1, axis=1) / np.linalg.norm(benchmark["forces"], ord=1, axis=1)),
        torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2)).item(),
        torch.sqrt(torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2))).item(),
        benchmark_time,
        np.sum(result['times']),
        np.mean(result['times'])
    ]

    return row

def run_mattersim_error_test(atoms, num_parts, num_mp):
    benchmark_start = time.perf_counter()
    benchmark = get_mattersim_benchmark(atoms)
    benchmark_end = time.perf_counter()

    mattersim_partition_inference.model_adapter.num_message_passing = num_mp

    experiment_start = time.perf_counter()
    result = mattersim_partition_inference.run(atoms, desired_partitions=NUM_PARTITIONS)
    experiment_end = time.perf_counter()

    torch.cuda.empty_cache()
    
    return {
        'num_atoms': len(atoms),
        'num_partitions': num_parts,
        
    }

    row = [
        len(atoms),
        num_parts,
        np.mean(result['partition_sizes']),
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
    ]
    
    return row
        
elements = ['C', 'H', 'O', 'N']

def create_random_atoms(num_atoms, density=0.1):
    volume = num_atoms / density
    box_size = volume**(1/3)
    
    positions = np.random.rand(num_atoms, 3) * box_size
    symbols = np.random.choice(elements, size=num_atoms)
    return Atoms(symbols=symbols, positions=positions, cell=[box_size, box_size, box_size], pbc=True)
        
for num_atoms in range(1000, 70000, 5000):
    suitable_mp = mp_list.copy()
    if num_atoms > 30000:
        suitable_mp = [mp for mp in mp_list if mp <= 5]  # Limit MP for large systems
    for mp in suitable_mp:
        for _ in range(5):
            atoms = create_random_atoms(num_atoms)

            orb_results = run_orb_error_test(atoms, NUM_PARTITIONS, mp)
            file_exists = os.path.isfile(ORB_RESULTS)
            with open(ORB_RESULTS, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(fields)
                writer.writerow(orb_results + [1])
            
            mattersim_results = run_mattersim_error_test(atoms, NUM_PARTITIONS, mp)
            file_exists = os.path.isfile(MATTERSIM_RESULTS)
            with open(MATTERSIM_RESULTS, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(fields)
                writer.writerow(mattersim_results + [1])

parser = argparse.ArgumentParser(description="Partitioned inference error test")
parser.add_argument("--results_dir", type=str, required=True, help="Directory for results")
parser.add_argument("--method", type=str, required=True, help="Partitioning method")
parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.xyz)")
parser.add_argument("--partitions", type=int, default=10, help="Number of partitions")
parser.add_argument("--message_passing", type=int, default=5, help="Number of message passing layers")

args = parser.parse_args()

atoms = read(args['dataset'])
partitions = args['partitions']
message_passing = args['message_passing']
results_dir = args['results_dir']

benchmark_start = time.perf_counter()
benchmark = get_mattersim_benchmark(atoms)
benchmark_end = time.perf_counter()

mattersim_partition_inference.model_adapter.num_message_passing = message_passing

experiment_start = time.perf_counter()
result = mattersim_partition_inference.run(atoms, desired_partitions=partitions)
experiment_end = time.perf_counter()

torch.cuda.empty_cache()

forces_benchmark = benchmark['forces']
forces_experiment = result['forces']


params = {
    'num_atoms': len(atoms),
    'num_partitions': partitions,
    'message_passing': message_passing
}