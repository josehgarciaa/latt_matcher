# latmatcher.bilayer
from __future__ import annotations
import numpy as np
from ase import Atoms
from .core import rotation_matrix, strain_matrix
from .cif import read_cif_atoms, lattice_xy_from_atoms


def apply_inplane_transform(atoms: Atoms, s1: float, s2: float, theta: float) -> Atoms:
    new = atoms.copy()
    S = strain_matrix(s1, s2)
    RT = rotation_matrix(theta).T
    T = S @ RT

    pos = new.get_positions()
    pos[:, :2] = pos[:, :2] @ T.T
    new.set_positions(pos)

    cell = np.array(new.cell)
    cell[0, :2] = T @ cell[0, :2]
    cell[1, :2] = T @ cell[1, :2]
    new.set_cell(cell, scale_atoms=False)
    return new


def set_cell_from_inplane_and_z(atoms: Atoms, A_xy: np.ndarray, z_length: float) -> None:
    a1 = np.array([A_xy[0, 0], A_xy[1, 0], 0.0])
    a2 = np.array([A_xy[0, 1], A_xy[1, 1], 0.0])
    a3 = np.array([0.0, 0.0, z_length])
    atoms.set_cell(np.vstack([a1, a2, a3]), scale_atoms=False)


def wrap_xy_into_cell(atoms: Atoms) -> None:
    sp = atoms.get_scaled_positions(wrap=False)
    sp[:, 0] = sp[:, 0] - np.floor(sp[:, 0])
    sp[:, 1] = sp[:, 1] - np.floor(sp[:, 1])
    atoms.set_scaled_positions(sp)


def build_bilayer_from_cifs(cif_A: str, cif_B: str, params_B,
                            interlayer_distance_A: float = 10.0,
                            vacuum_extra_A: float = 10.0) -> Atoms:
    A = read_cif_atoms(cif_A)
    B = read_cif_atoms(cif_B)
    s1, s2, theta = params_B
    B_t = apply_inplane_transform(B, s1, s2, theta)
    A_xy = lattice_xy_from_atoms(A)

    zA = A.get_positions()[:, 2]
    zB = B_t.get_positions()[:, 2]
    thickA = float(zA.max() - zA.min()) if len(zA) else 0.0
    thickB = float(zB.max() - zB.min()) if len(zB) else 0.0
    z_len = thickA + thickB + interlayer_distance_A + vacuum_extra_A

    set_cell_from_inplane_and_z(A, A_xy, z_len)
    set_cell_from_inplane_and_z(B_t, A_xy, z_len)

    z0 = 0.5 * vacuum_extra_A
    pA = A.get_positions()
    pA[:, 2] += (z0 - pA[:, 2].min())
    A.set_positions(pA)

    pB = B_t.get_positions()
    pB[:, 2] += ((z0 + thickA + interlayer_distance_A) - pB[:, 2].min())
    B_t.set_positions(pB)

    wrap_xy_into_cell(A)
    wrap_xy_into_cell(B_t)

    combined = A + B_t
    set_cell_from_inplane_and_z(combined, A_xy, z_len)
    return combined
