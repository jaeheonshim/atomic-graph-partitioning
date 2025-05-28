from .base import GraphPartitioner
import metis
import ase
import networkx as nx
from metis_cython.partition import part_graph_kway_extended

class MetisCythonPartitioner(GraphPartitioner):
    def partition(self, atoms, adj_list, desired_partitions, mp = 1):
        result = part_graph_kway_extended(adj_list, desired_partitions, distance=mp)
        
        return result