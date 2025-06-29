from main.inference import AtomicPartitionInference
from main.implementations.mattersim import MatterSimModelAdapter
from main.implementations.orb import OrbModelAdapter
from main.partitioner.metis_cython import MetisCythonPartitioner

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
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("MP_API_KEY")

def get_mp_atoms(id):
    with MPRester(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(id)
        
        atoms = AseAtomsAdaptor.get_atoms(structure)
        
        return atoms
    
MATTERSIM_RESULTS = "results/test/mattersim_results_randomized.csv"
ORB_RESULTS = "results/test/orb_results_randomized.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

ATOMS_FILE = "datasets/test.xyz"
NUM_PARTITIONS = 60
mp_list = [2,3,4,5,6,7,8]
box_sizes = [10, 25, 50, 100]

MATTERSIM_ITERATIONS = 1 # Mattersim is a little weird so I will run multiple times and average

orbff = pretrained.orb_v2(device=device)

orb_partition_inference = AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=3), partitioner=MetisCythonPartitioner())
mattersim_partition_inference = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=3), partitioner=MetisCythonPartitioner())

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partitioned inference error test")
    parser.add_argument("--mp_id", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['mattersim', 'orb'], required=True)
    parser.add_argument("--num_atoms", type=int, required=True)
    parser.add_argument("--num_partitions", type=int, required=True)
    parser.add_argument("--message_passing", type=int, default=5, help="Number of message passing layers")
    parser.add_argument("--result_dir", type=str, required=True)

    args = parser.parse_args()
    
    id = args.mp_id
    num_atoms = args.num_atoms
    desired_partitions = args.num_partitions
    message_passing = args.message_passing
    
    atoms = get_mp_atoms(id)
    scale = int((num_atoms / len(atoms)) ** (1/3)) + 1
    atoms = make_supercell(atoms, ((scale, 0, 0), (0, scale, 0), (0, 0, scale)))

    benchmark_start = time.perf_counter()
    benchmark = get_mattersim_benchmark(atoms)
    benchmark_end = time.perf_counter()

    mattersim_partition_inference.model_adapter.num_message_passing = message_passing

    experiment_start = time.perf_counter()
    result = mattersim_partition_inference.run(atoms, desired_partitions=desired_partitions)
    experiment_end = time.perf_counter()

    torch.cuda.empty_cache()
    
    energy_benchmark = benchmark['energy']
    energy_experiment = result['energy']

    forces_benchmark = benchmark['forces']
    forces_experiment = result['forces']
    
    metadata = {
        'mp_id': id,
        'desired_partitions': desired_partitions,
        'num_atoms': len(atoms),
        'message_passing': message_passing,
        'benchmark_time': benchmark_end - benchmark_start,
        'experiment_time': experiment_end - experiment_start
    }
    
    np.savez(
        f'{id}_{len(atoms)}_{desired_partitions}_{message_passing}',
        metadata=metadata,
        energy_benchmark=energy_benchmark,
        energy_experiment=energy_experiment,
        forces_benchmark=forces_benchmark,
        forces_experiment=forces_experiment
    )