from adapter import AtomicModelAdapter

import ase
import networkx as nx
import torch

from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from networkx import Graph
from orb_models.forcefield.base import AtomGraphs

from orb_models.forcefield import segment_ops

from orb_models.forcefield.pretrained import get_base, load_model_for_inference
from orb_models.forcefield.graph_regressor import GraphRegressor, EnergyHead, NodeHead, GraphHead

from orb_models.forcefield.graph_regressor import ScalarNormalizer, LinearReferenceEnergy
from orb_models.forcefield.reference_energies import REFERENCE_ENERGIES


class OrbModelAdapter(AtomicModelAdapter[AtomGraphs]):
    def __init__(self, *args, **kwargs):
        super().__init__(
            embedding_size=256,
            *args, **kwargs
        )

        ref = REFERENCE_ENERGIES["vasp-shifted"]
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=True
        )

        base = get_base(num_message_passing_steps=4)

        model = GraphRegressor(
            graph_head=EnergyHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                target="energy",
                node_aggregation="mean",
                reference_energy_name="vasp-shifted",
                train_reference=True,
                predict_atom_avg=True,
            ),
            node_head=NodeHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                target="forces",
                remove_mean=True,
            ),
            stress_head=GraphHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                target="stress",
                compute_stress=True,
            ),
            model=base,
        )

        self.orbff = load_model_for_inference(model, 'https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt', self.device)

    def atoms_to_graph(self, atoms):
        return ase_atoms_to_atom_graphs(atoms, device=self.device)

    def graph_to_networkx(self, graph):
        senders = graph.senders
        receivers = graph.receivers
        edge_feats = graph.edge_features

        G = nx.Graph()
        G.add_nodes_from(range(graph.n_node))

        for i, u in enumerate(senders):
            G.add_edge(u.item(), receivers[i].item(), weight=edge_feats['r'])

        return G

    def forward_graph(self, graph):
        batch = self.orbff.model(graph)
        return batch.node_features["feat"] 
    
    def predict_energy(self, embeddings, atoms):
        n_node = torch.tensor([embeddings.shape[0]])

        input = segment_ops.aggregate_nodes(
            embeddings,
            n_node,
            reduction="mean"
        )

        energy = self.orbff.graph_head.mlp(input)
        energy = self.orbff.graph_head.normalizer.inverse(energy).squeeze(-1)
        energy = energy * n_node
        energy = energy + self.reference(torch.tensor(atoms.get_atomic_numbers()), n_node)
        
        return energy