from ..adapter import AtomicModelAdapter

from typing import Dict, List, Optional

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from mattersim.datasets.utils.convertor import GraphConvertor
from mattersim.forcefield.potential import batch_to_dict

class MatterSimModelAdapter(AtomicModelAdapter[Data]):
    def __init__(self, *args, **kwargs):
        super().__init__(
            embedding_size=128,
            # num_message_passing=5,
            *args, **kwargs
        )

        self.converter = GraphConvertor("m3gnet", 5.0, True, 4.0)
        self.model = _load_modified_from_checkpoint(device=self.device)

    def atoms_to_graph(self, atoms):
        return self.converter.convert(atoms, None, None, None)

    def graph_to_adjlist(self, graph):
        edge_index = graph.edge_index

        num_nodes = graph.num_nodes
        adjlist = [[] for _ in range(num_nodes)]

        for src, dst in edge_index.t().tolist():
            adjlist[src].append(dst)
            adjlist[dst].append(src)

        return adjlist
    
    def init_partition(self, all_atoms, partitions, roots):
        super().init_partition(all_atoms, partitions, roots)

        self.total_energy = 0
        self.forces = torch.zeros((len(all_atoms), 3), dtype=torch.float32, device=self.device)
        self.atomic_numbers = torch.tensor(all_atoms.get_atomic_numbers()).long()

    def forward_graph(self, graphs, part_indices):
        dataloader = DataLoader(graphs)

        embeddings = []
        for i, input_graph in enumerate(dataloader):
            input_graph = input_graph.to(self.device)
            input_dict = batch_to_dict(input_graph)
            input_dict['atom_pos'].requires_grad_(True)

            result = self.model.forward(input_dict)

            ## Energy
            energy = self.model.final(result).view(-1)
            energy = self.model.normalizer(energy, self.atomic_numbers[self.partitions[part_indices[i]]])
            forces = self._compute_forces_from_grad(input_dict['atom_pos'], energy)
            
            for j in range(len(self.partitions[part_indices[i]])):
                if self.roots[part_indices[i]][j]:
                    self.total_energy += energy[j]
                    self.forces[self.partitions[part_indices[i]][j]] = forces[j]

            embeddings.append(result)

        return embeddings
    
    def _compute_forces_from_grad(self, atom_pos, energy):
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(
                energy,
            )
        ]
        
        grad = torch.autograd.grad(
            outputs=[
                energy,
            ],
            inputs=[atom_pos],
            grad_outputs=grad_outputs,
            create_graph=False,
        )

        # Dump out gradient for forces
        force_grad = grad[0]
        if force_grad is not None:
            forces = torch.neg(force_grad)
            
            return forces
    
    def predict_energy(self, embeddings, atoms):
        return self.total_energy
    
    def predict_forces(self, embeddings, atoms):
        return self.forces


### Patched implementations of MatterSim classes

from typing import Dict

import os

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from torch_runstats.scatter import scatter
from mattersim.utils.download_utils import download_checkpoint

class _M3GnetModified(M3Gnet):
    def forward(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].float()
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        # -------------------------------------------------------------#
        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        index_bias = torch.repeat_interleave(  # noqa: F501
            cumsum, num_three_body, dim=0
        ).unsqueeze(-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = torch.repeat_interleave(repeats=num_atoms)
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop
        for idx, conv in enumerate(self.graph_conv):
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        return atom_attr  # [batch_size]
    
def _load_modified_from_checkpoint(
    load_path: str = None,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    checkpoint_folder = os.path.expanduser("~/.local/mattersim/pretrained_models")
    os.makedirs(checkpoint_folder, exist_ok=True)
    if (
        load_path is None
        or load_path.lower() == "mattersim-v1.0.0-1m.pth"
        or load_path.lower() == "mattersim-v1.0.0-1m"
    ):
        load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-1M.pth")
        if not os.path.exists(load_path):
            print(
                "The pre-trained model is not found locally, "
                "attempting to download it from the server."
            )
            download_checkpoint(
                "mattersim-v1.0.0-1M.pth", save_folder=checkpoint_folder
            )
        print(f"Loading the pre-trained {os.path.basename(load_path)} model")
    elif (
        load_path.lower() == "mattersim-v1.0.0-5m.pth"
        or load_path.lower() == "mattersim-v1.0.0-5m"
    ):
        load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-5M.pth")
        if not os.path.exists(load_path):
            print(
                "The pre-trained model is not found locally, "
                "attempting to download it from the server."
            )
            download_checkpoint(
                "mattersim-v1.0.0-5M.pth", save_folder=checkpoint_folder
            )
        print(f"Loading the pre-trained {os.path.basename(load_path)} model")
    else:
        print("Loading the model from %s" % load_path)
    assert os.path.exists(load_path), f"Model file {load_path} not found"

    checkpoint = torch.load(load_path, map_location=device)

    model = _M3GnetModified(device=device, **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)

    model.eval()

    return model