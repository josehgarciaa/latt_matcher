import numpy as np
from latmatcher.core import rotation_matrix, strain_matrix, make_index_set, coverage_and_grad, hex_lattice


def test_rotation_matrix_orthonormal():
    R = rotation_matrix(0.37)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-12)


def test_strain_matrix_diagonal():
    assert np.allclose(strain_matrix(0.1, -0.2), np.array([[1.1, 0.0], [0.0, 0.8]]))


def test_coverage_grad_shapes():
    A = hex_lattice(1.0)
    B = hex_lattice(1.01)
    m = make_index_set(7, 7)
    C, g = coverage_and_grad(np.array([0.0, 0.0, 0.0]), A, B, m)
    assert np.isscalar(C)
    assert g.shape == (3,)
