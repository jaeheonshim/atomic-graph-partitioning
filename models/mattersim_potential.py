import torch
from mattersim.forcefield.potential import Potential
from ase.units import GPa

class PartitionPotential(Potential):
    def forward(
        self,
        input: dict[str, torch.Tensor],
        include_forces: bool = True,
        include_stresses: bool = True,
        dataset_idx: int = -1,
    ):
        output = {}
        if self.model_name == "graphormer" or self.model_name == "geomformer":
            raise NotImplementedError
        else:
            strain = torch.zeros_like(input["cell"], device=self.device)
            volume = torch.linalg.det(input["cell"])
            if include_forces is True:
                input["atom_pos"].requires_grad_(True)
            if include_stresses is True:
                strain.requires_grad_(True)
                input["cell"] = torch.matmul(
                    input["cell"],
                    (torch.eye(3, device=self.device)[None, ...] + strain),
                )
                strain_augment = torch.repeat_interleave(
                    strain, input["num_atoms"], dim=0
                )
                input["atom_pos"] = torch.einsum(
                    "bi, bij -> bj",
                    input["atom_pos"],
                    (torch.eye(3, device=self.device)[None, ...] + strain_augment),
                )
                volume = torch.linalg.det(input["cell"])

            energies, energies_i = self.model.forward(input, dataset_idx, return_energies_per_atom=True)
            output["energies"] = energies
            output["energies_i"] = energies_i

            # Only take first derivative if only force is required
            if include_forces is True and include_stresses is False:
                grad_outputs: list[torch.Tensor] = [
                    torch.ones_like(
                        energies,
                    )
                ]
                grad = torch.autograd.grad(
                    outputs=[
                        energies,
                    ],
                    inputs=[input["atom_pos"]],
                    grad_outputs=grad_outputs,
                    create_graph=self.model.training,
                )

                # Dump out gradient for forces
                force_grad = grad[0]
                if force_grad is not None:
                    forces = torch.neg(force_grad)
                    output["forces"] = forces

            # Take derivatives up to second order
            # if both forces and stresses are required
            if include_forces is True and include_stresses is True:
                grad_outputs: list[torch.Tensor] = [
                    torch.ones_like(
                        energies,
                    )
                ]
                grad = torch.autograd.grad(
                    outputs=[
                        energies,
                    ],
                    inputs=[input["atom_pos"], strain],
                    grad_outputs=grad_outputs,
                    create_graph=self.model.training,
                )

                # Dump out gradient for forces and stresses
                force_grad = grad[0]
                stress_grad = grad[1]

                if force_grad is not None:
                    forces = torch.neg(force_grad)
                    output["forces"] = forces

                if stress_grad is not None:
                    stresses = (
                        1 / volume[:, None, None] * stress_grad / GPa
                    )  # 1/GPa = 160.21766208
                    output["stresses"] = stresses

        return output