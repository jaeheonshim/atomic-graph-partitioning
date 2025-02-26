from wrapper.inference import AtomicPartitionInference
from wrapper.implementations.mattersim import MatterSimModelAdapter
from wrapper.implementations.orb import OrbModelAdapter

from ase.io import read
from ase.build import make_supercell

from mattersim.forcefield import MatterSimCalculator

from orb_models.forcefield import atomic_system, pretrained

import torch

device = 'cpu'

ATOMS_FILE = "datasets/H2O.xyz"
MAX_SUPERCELL_DIM = 6
NUM_PARTITIONS = 10

orbff = pretrained.orb_v2(device=device)
mattersim_calc = MatterSimCalculator()

orb_part_inference = [
    AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=4)),
    AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=5)),
    AtomicPartitionInference(OrbModelAdapter(device=device, num_message_passing=6))
]

# mattersim_mp_4 = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=4))
# mattersim_mp_5 = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=5))
# mattersim_mp_6 = AtomicPartitionInference(MatterSimModelAdapter(device=device, num_message_passing=6))

fields = ['num_atoms', 'mp', 'energy_error_abs', 'energy_error_pct', 'forces_error_max', 'forces_error_mae', 'forces_error_mape', 'forces_error_mse', 'forces_error_rms']
orb_rows = []
mattersim_rows = []

def get_mattersim_benchmark(atoms):
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

    for inference in orb_part_inference:
        mp = inference.model_adapter.num_message_passing
        result = inference.run(atoms, desired_partitions=NUM_PARTITIONS)
        
        orb_rows.append([
            len(atoms),
            mp,
            abs(benchmark["energy"] - result["energy"]).item(),
            abs((benchmark["energy"] - result["energy"]) / benchmark["energy"]).item(),
            torch.max(torch.abs(benchmark["forces"] - result["forces"])).item(),
            torch.mean(torch.abs(benchmark["forces"] - result["forces"])).item(),
            torch.mean(torch.abs((benchmark["forces"] - result["forces"]) / benchmark["forces"])).item(),
            torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2)).item(),
            torch.sqrt(torch.mean(torch.pow(benchmark["forces"] - result["forces"], 2))).item(),
        ])
        

for x in range(1, MAX_SUPERCELL_DIM):
    run_orb_error_test(((x, 0, 0), (0, x, 0), (0, 0, x)))
    print(orb_rows)