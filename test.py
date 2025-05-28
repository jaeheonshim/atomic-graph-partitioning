import networkx as nx
import cProfile
import random
import timeit
import matplotlib.pyplot as plt

from partitioning.adjlist_partitioner import part_graph_extended
from main.partitioner.metis_cython import MetisCythonPartitioner

random.seed(4)

G = nx.erdos_renyi_graph(n=100, p=0.3)

p = MetisCythonPartitioner()

adjlist = [list(G.neighbors(node)) for node in sorted(G.nodes())]

result = p.partition(None, adjlist, 5, 2)

print(result)