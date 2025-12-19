import numpy as np
from latmatcher.core import hex_lattice, make_index_set
from latmatcher.optim import maximise_coverage


def test_optim_scaled_half_pushes_strain_positive():
    A = hex_lattice(1.0)
    B = hex_lattice(0.5)
    m = make_index_set(7, 7)
    bounds = [(-0.2, 2.0), (-0.2, 2.0), (-np.pi/6, np.pi/6)]
    xopt, Copt, info = maximise_coverage(A, B, m, bounds=bounds, x0=np.array([0.0, 0.0, 0.0]),
                                         sigma=0.12, M=10, P=3)
    assert xopt[0] > 0.2 and xopt[1] > 0.2 and info["success"] is True


def test_optim_rotated_stays_near_initial_target():
    A = hex_lattice(1.0)
    B = hex_lattice(1.0)
    m = make_index_set(9, 9)
    theta_target = 2.0 * np.pi / 180.0
    bounds = [(-0.05, 0.05), (-0.05, 0.05),
              (theta_target - 5*np.pi/180, theta_target + 5*np.pi/180)]
    xopt, Copt, info = maximise_coverage(A, B, m, bounds=bounds, x0=np.array([0.0, 0.0, theta_target]),
                                         sigma=0.12, M=10, P=3)
    assert abs(xopt[2] - theta_target) < 1.0 * np.pi / 180.0 and info["success"] is True
