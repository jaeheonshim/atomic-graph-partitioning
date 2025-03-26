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
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

ATOMS_FILE = "datasets/test.xyz"

mattersim_partition_inference = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=4))


def get_mattersim_benchmark(atoms):
    mattersim_calc = MatterSimCalculator(compute_stress=False)
    atoms.calc = mattersim_calc

    return {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces()
    }

       
atoms = read(ATOMS_FILE)
atoms = make_supercell(atoms, ((2, 0, 0), (0, 2, 0), (0, 0, 2)))
result = mattersim_partition_inference.run(atoms, desired_partitions=60)
benchmark = get_mattersim_benchmark(atoms)

benchmark_forces = benchmark['forces']
result_forces = result['forces']

print(benchmark_forces)
print(result_forces)

print('Max error', np.max(np.abs(benchmark_forces - result_forces)))