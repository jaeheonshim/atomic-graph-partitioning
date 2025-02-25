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

    def forward_graph(self, graph: GraphType) -> torch.Tensor:
        """
        Model specific graph through graph regressor for embeddigs
        """
        ...
    
    def forward_energy(self, embeddings: torch.Tensor, atoms: ase.Atoms) -> torch.Tensor:
        ...