from typing import Protocol, TypeVar, Generic, List, Any

import ase
import networkx as nx
import torch

GraphType = TypeVar("GraphType")

class AtomicModelAdapter(Generic[GraphType]):
    def __init__(
        self,
        *,
        embedding_size: int,
        num_message_passing: int,
        device: torch.device = "cpu"
    ):
        self.device = device
        self.embedding_size = embedding_size
        self.num_message_passing = num_message_passing

    def atoms_to_graph(self, atoms: ase.Atoms) -> GraphType:
        """
        Model specific conversion from ase.Atoms to graph
        """
        ...

    def graph_to_adjlist(self, graph: GraphType) -> list[list[int]]:
        """
        Model specific graph to networkX graph
        """
        ...

    def init_partition(self, all_atoms: ase.Atoms, partitions: list[list[int]], roots: list[list[bool]]):
        """
        Store information about the partition configuration so that it can be used in other methods
        """
        self.all_atoms = all_atoms
        self.partitions = partitions
        self.roots = roots

    def forward_graph(self, graphs: list[GraphType], part_indices: list[int]) -> list[any]:
        """
        Model specific graph through graph regressor for embeddigs
        """
        ...
    
    def predict_energy(self, embeddings: torch.Tensor, atoms: ase.Atoms) -> torch.Tensor:
        ...

    def predict_forces(self, embeddings: torch.Tensor, atoms: ase.Atoms) -> torch.Tensor:
        ...