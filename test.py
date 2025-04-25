import networkx as nx
import cProfile
import random
import timeit
import matplotlib.pyplot as plt

from partitioning.adjlist_partitioner import part_graph_extended
from metis_wrapper.partition import part_graph_kway_extended as optim_part

random.seed(42)

def networkx_to_metis_adjlist(G):    
    n = G.number_of_nodes()
    adjlist = [[] for _ in range(n)]
    
    for u, v, data in G.edges(data=True):
        adjlist[u].append(v)

    return adjlist

def create_random_graph(n, p):
    G = nx.erdos_renyi_graph(n=n, p=p)
        
    return G


def benchmark_optim_one_trial(n, p, parts, dist):
    G = create_random_graph(n, p)
    adjlist = networkx_to_metis_adjlist(G)
    
    benchmark = part_graph_extended(adjlist, parts, dist)
    
    start = timeit.default_timer()
    result = optim_part(adjlist, parts, distance=dist)
    print(benchmark[0])
    print(result)
    stop = timeit.default_timer()
    
    return stop - start

# %% [markdown]
# Profiling

# %%
# def profile_run():
#     benchmark_adjlist_one_trial(10000, 0.3, 5, 4)

# cProfile.run('profile_run()', 'adj_part.prof')

# %%
sizes = []
optim_time = []

size = 64

while True:
    time3 = benchmark_optim_one_trial(size, 0.3, 5, 4)
    
    sizes.append(size)
    optim_time.append(time3)
    
    print(size, time3)
    
    size *= 2

