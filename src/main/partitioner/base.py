from abc import ABC, abstractmethod
from ase import Atoms
import networkx as nx

class GraphPartitioner(ABC):
    @abstractmethod
    def partition(self, atoms: Atoms, adj_list: nx.Graph, desired_partitions: int, mp: int = 1) -> tuple[list[set], list[set]]:
        """Partition a graph and return (partitions, extended_partitions)"""
        pass