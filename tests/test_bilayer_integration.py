import numpy as np
from ase import Atoms
from latmatcher.cif import write_cif_atoms
from latmatcher.bilayer import build_bilayer_from_cifs


def make_layer(cell_xy=(2.0, 2.0), z=10.0):
    cell = np.array([[cell_xy[0], 0.0, 0.0],
                     [0.0, cell_xy[1], 0.0],
                     [0.0, 0.0, z]])
    return Atoms(symbols=["C", "C"],
                 positions=[[0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0]],
                 cell=cell, pbc=True)


def test_build_bilayer_from_two_cifs(tmp_path):
    A = make_layer(); B = make_layer()
    cifA = tmp_path / "A.cif"; cifB = tmp_path / "B.cif"
    write_cif_atoms(A, str(cifA)); write_cif_atoms(B, str(cifB))
    combined = build_bilayer_from_cifs(str(cifA), str(cifB), (0.0, 0.0, 0.0))
    assert len(combined) == len(A) + len(B)
