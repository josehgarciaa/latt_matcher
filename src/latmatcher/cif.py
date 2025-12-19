# latmatcher.cif
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.io import read, write


def read_cif_atoms(path: str) -> Atoms:
    return read(path)


def write_cif_atoms(atoms: Atoms, out_path: str) -> None:
    write(out_path, atoms, format="cif")


def lattice_xy_from_atoms(atoms: Atoms) -> np.ndarray:
    cell = np.array(atoms.cell)
    return np.column_stack([cell[0, :2], cell[1, :2]]).astype(float)
