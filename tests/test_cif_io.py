import numpy as np
from ase import Atoms
from latmatcher.cif import write_cif_atoms, read_cif_atoms


def test_cif_read_write_roundtrip(tmp_path):
    cell = np.array([[2.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, 10.0]])
    atoms = Atoms(symbols=["C", "C"],
                  positions=[[0.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0]],
                  cell=cell, pbc=True)
    out = tmp_path / "test.cif"
    write_cif_atoms(atoms, str(out))
    atoms2 = read_cif_atoms(str(out))
    assert np.allclose(np.array(atoms2.cell), cell, atol=1e-6)
