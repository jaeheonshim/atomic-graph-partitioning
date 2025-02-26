from typing import Protocol, TypeVar, Generic, List, Any

import ase
import networkx as nx
import torch

GraphType = TypeVar("GraphType")

class AtomicModelAdapter(Generic[GraphType]):
    def __init__(self, *, device: torch.device = "cpu", embedding_size: int):
        self.device = device
        self.embedding_size = embedding_size

    def atoms_to_graph(self, atoms: ase.Atoms) -> GraphType:
        """
        Model specific conversion from ase.Atoms to graph
        """
        ...

    def graph_to_networkx(self, graph: GraphType) -> nx.Graph:
        """
        Model specific graph to networkX graph
        """
        ...

    def set_partition_info(self, all_atoms: ase.Atoms, partitions: torch.Tensor, roots: torch.Tensor):
        """
        Store information about the partition configuration so that it can be used in other methods
        """
        self.all_atoms = all_atoms
        self.partitions = partitions
        self.roots = roots

    def forward_graph(self, graph: GraphType, part_index: int) -> torch.Tensor:
        """
        Model specific graph through graph regressor for embeddigs
        """
        ...
    
    def predict_energy(self, embeddings: torch.Tensor, atoms: ase.Atoms) -> torch.Tensor:
        ...

    def predict_forces(self, embeddings: torch.Tensor, atoms: ase.Atoms) -> torch.Tensor:
        ...