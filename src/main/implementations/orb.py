from ..adapter import AtomicModelAdapter

import ase
import networkx as nx
import torch

from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.base import batch_graphs

from orb_models.forcefield import segment_ops

from orb_models.forcefield.pretrained import get_base, load_model_for_inference
from orb_models.forcefield.graph_regressor import GraphRegressor, EnergyHead, NodeHead, GraphHead

from orb_models.forcefield.graph_regressor import ScalarNormalizer, LinearReferenceEnergy
from orb_models.forcefield.reference_energies import REFERENCE_ENERGIES

class OrbModelAdapter(AtomicModelAdapter[AtomGraphs]):
    def __init__(self, *args, **kwargs):
        super().__init__(
            embedding_size=256,
            # num_message_passing=4,
            *args, **kwargs
        )

        ref = REFERENCE_ENERGIES["vasp-shifted"]
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=True
        ).to(self.device)

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

    def forward_graph(self, graphs, part_indices):
        batch = self.orbff.model(batch_graphs(graphs))
        node_feats = batch.node_features["feat"]

        embeddings = []
        i = 0
        j = 0
        while i < len(graphs):
            embeddings.append(node_feats[j:j+graphs[i].n_node.item()])
            j += graphs[i].n_node.item()
            i += 1

        return embeddings
    
    def predict_energy(self, embeddings, atoms):
        n_node = torch.tensor([embeddings.shape[0]], device=self.device)

        input = segment_ops.aggregate_nodes(
            embeddings,
            n_node,
            reduction="mean"
        )

        energy = self.orbff.graph_head.mlp(input)
        energy = self.orbff.graph_head.normalizer.inverse(energy).squeeze(-1)
        energy = energy * n_node
        energy = energy + self.reference(torch.tensor(atoms.get_atomic_numbers(), device=self.device), n_node)
        
        return energy
    
    def predict_forces(self, embeddings, atoms):
        n_node = torch.tensor([embeddings.shape[0]], device=self.device)

        forces = self.orbff.node_head.mlp(embeddings)
        system_means = segment_ops.aggregate_nodes(
            forces, n_node, reduction="mean"
        )
        node_broadcasted_means = torch.repeat_interleave(
            system_means, n_node, dim=0
        )
        forces = forces - node_broadcasted_means
        forces = self.orbff.node_head.normalizer.inverse(forces)
        
        return forces