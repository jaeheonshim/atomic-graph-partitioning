import random
from collections import deque

def descendants_at_distance_multisource(adj_list, sources, distance=None):
    queue = deque(sources)
    depths = deque([0 for _ in queue])
    visited = set(sources)

    while queue:
        node = queue[0]
        depth = depths[0]

        if distance is not None and depth > distance: return

        yield queue[0]

        queue.popleft()
        depths.popleft()

        for adj in adj_list[node]:
            if adj not in visited:
                visited.add(adj)
                queue.append(adj)
                depths.append(depth + 1)

def erdos_renyi_adjlist(n, p):
    adjlist = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):           
            if random.random() < p:
                adjlist[i].append(j)
                adjlist[j].append(i)
                
    return adjlist

import numpy as np
from ase import Atoms

def random_atoms_from_adjlist(
        adjlist,
        box_size=10.0,
        element_pool=("C",),
        seed=None,
        pbc=True):
    if seed is not None:
        np.random.seed(seed)

    n_atoms = len(adjlist)
    symbols  = np.random.choice(element_pool, size=n_atoms)
    positions = np.random.random((n_atoms, 3)) * box_size

    atoms = Atoms(symbols=symbols,
                  positions=positions,
                  cell=[box_size, box_size, box_size],
                  pbc=pbc)

    return atoms
