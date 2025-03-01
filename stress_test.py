from ase import Atoms
from ase.io import read
import numpy as np
import copy


def virial_stress(atoms: Atoms, forces: np.ndarray) -> np.ndarray:
    stress = np.zeros((3, 3))
    for i, force in enumerate(forces):
        stress += np.outer(atoms.positions[i], force)
    volume = atoms.get_volume()
    stress /= volume
    return stress


atoms_list: list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/mptraj_val.xyz", index=":") # type: ignore

for atoms in atoms_list:
    forces = np.array(atoms.get_forces())
    stress = np.array(atoms.get_stress(voigt=False))
    virial = virial_stress(copy.deepcopy(atoms), forces)
    assert np.allclose(stress, virial), f"Stress mismatch: {stress} vs {virial}"
print("Passed")