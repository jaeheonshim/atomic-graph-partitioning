import argparse
import os
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell

from mattersim.datasets.utils.convertor import GraphConvertor
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs

from main.partitioner import PARTITIONERS

import time

from filelock import FileLock
import json

from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("MP_API_KEY")

mattersim_converter = GraphConvertor("m3gnet", 5.0, True, 4.0)

def atoms_to_adjlist_orb(atoms):
    graph = ase_atoms_to_atom_graphs(atoms)
    senders = graph.senders
    receivers = graph.receivers
    
    adjlist = [[] for _ in range(graph.n_node)]
    
    for i, u in enumerate(senders):
        v = receivers[i]
        
        adjlist[u.item()].append(v.item())
        adjlist[v.item()].append(u.item())

    return adjlist

def atoms_to_adjlist_mattersim(atoms):
    data = mattersim_converter.convert(atoms, None, None, None)
    edge_index = data.edge_index  # shape: [2, num_edges]

    num_nodes = data.num_nodes
    adjlist = [[] for _ in range(num_nodes)]

    for src, dst in edge_index.t().tolist():
        adjlist[src].append(dst)
        adjlist[dst].append(src)

    return adjlist

def write_result(result_file, record):
    lock = FileLock(result_file + ".lock")
    with lock:
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")

def run_orb_benchmark(id, atoms, atoms_to_adjlist, partitioner_name, desired_partitions, mp):
    partitioner = PARTITIONERS[partitioner_name]()
    adjlist = atoms_to_adjlist(atoms)
        
    start = time.perf_counter()
    core, extended = partitioner.partition(atoms, adjlist, desired_partitions, mp)
    end = time.perf_counter()
    
    return {
        'mp_id': id,
        'partitioner': partitioner_name,
        'desired_partitions': desired_partitions,
        'actual_partitions': len(core),
        'core': [list(a) for a in core],
        'extended': [list(a) for a in extended],
        'time': end - start
    }

def get_mp_atoms(id):
    with MPRester(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(id)
        
        atoms = AseAtomsAdaptor.get_atoms(structure)
        
        return atoms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_id", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['mattersim', 'orb'], required=True)
    parser.add_argument("--partitioner", type=str, choices=PARTITIONERS, required=True)
    parser.add_argument("--num_atoms", type=int, required=True)
    parser.add_argument("--num_partitions", type=int, required=True)
    parser.add_argument("--result_file", type=str, required=True)

    args = parser.parse_args()
    
    id = args.mp_id
    model = args.model
    partitioner = PARTITIONERS[args.partitioner]
    num_atoms = args.num_atoms
    desired_partitions = args.num_partitions
    result_file = args.result_file
    
    atoms = get_mp_atoms(id)
    scale = int((args.num_atoms / len(atoms)) ** (1/3)) + 1
    atoms = make_supercell(atoms, ((scale, 0, 0), (0, scale, 0), (0, 0, scale)))
    
    if model == 'orb':
        result = run_orb_benchmark(id, atoms, atoms_to_adjlist_orb, args.partitioner, desired_partitions, 5)
        result['model'] = 'orb'
    elif model == 'mattersim':
        result = run_orb_benchmark(id, atoms, atoms_to_adjlist_mattersim, args.partitioner, desired_partitions, 5)
        result['model'] = 'mattersim'
        
    write_result(result_file, result)