# latmatcher.core
# Core maths for 2D lattice matching (coverage + gradients)

from __future__ import annotations
import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def d_rotation_T_dtheta(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s,  c], [-c, -s]], dtype=float)


def strain_matrix(s1: float, s2: float) -> np.ndarray:
    return np.array([[1.0 + s1, 0.0], [0.0, 1.0 + s2]], dtype=float)


def make_index_set(N1: int, N2: int) -> np.ndarray:
    m1 = np.arange(N1, dtype=int)
    m2 = np.arange(N2, dtype=int)
    grid = np.stack(np.meshgrid(m1, m2, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def make_integer_grid(P: int) -> np.ndarray:
    r = np.arange(-P, P + 1, dtype=int)
    grid = np.stack(np.meshgrid(r, r, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def frac_fourier(x: np.ndarray, M: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.arange(1, M + 1, dtype=float)[:, None]
    return 0.5 - (1.0 / np.pi) * np.sum((1.0 / k) * np.sin(2.0 * np.pi * k * x[None, :]), axis=0)


def d_frac_fourier_dx(x: np.ndarray, M: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.arange(1, M + 1, dtype=float)[:, None]
    return -2.0 * np.sum(np.cos(2.0 * np.pi * k * x[None, :]), axis=0)


def F_periodic_gaussian(delta: np.ndarray, sigma: float, p_grid: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    diff = p_grid[None, :, :] - delta[:, None, :]
    r2 = np.sum(diff * diff, axis=-1)
    return np.sum(np.exp(-r2 / (4.0 * sigma * sigma)), axis=1)


def grad_F_wrt_delta(delta: np.ndarray, sigma: float, p_grid: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    diff = p_grid[None, :, :] - delta[:, None, :]
    r2 = np.sum(diff * diff, axis=-1)
    w = np.exp(-r2 / (4.0 * sigma * sigma))
    return np.sum(w[:, :, None] * diff, axis=1) / (2.0 * sigma * sigma)


def coverage_and_grad(params: np.ndarray, A: np.ndarray, B: np.ndarray, m_set: np.ndarray,
                      sigma: float = 0.12, M: int = 12, P: int = 4) -> tuple[float, np.ndarray]:
    s1, s2, theta = float(params[0]), float(params[1]), float(params[2])

    Ainv = np.linalg.inv(A)
    S = strain_matrix(s1, s2)
    RT = rotation_matrix(theta).T
    T = Ainv @ S @ B @ RT
    u = (T @ m_set.T).T

    delta = np.empty_like(u)
    delta[:, 0] = frac_fourier(u[:, 0], M)
    delta[:, 1] = frac_fourier(u[:, 1], M)

    p_grid = make_integer_grid(P)
    Fv = F_periodic_gaussian(delta, sigma, p_grid)
    const = 2.0 * np.pi * sigma * sigma
    C = const * np.sum(Fv)

    dF_ddelta = grad_F_wrt_delta(delta, sigma, p_grid)
    g0 = d_frac_fourier_dx(u[:, 0], M)
    g1 = d_frac_fourier_dx(u[:, 1], M)

    dS1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    dS2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)

    du_ds1 = (Ainv @ dS1 @ B @ RT @ m_set.T).T
    du_ds2 = (Ainv @ dS2 @ B @ RT @ m_set.T).T
    dRT = d_rotation_T_dtheta(theta)
    du_dth = (Ainv @ S @ B @ dRT @ m_set.T).T

    ddelta_ds1 = np.column_stack([g0 * du_ds1[:, 0], g1 * du_ds1[:, 1]])
    ddelta_ds2 = np.column_stack([g0 * du_ds2[:, 0], g1 * du_ds2[:, 1]])
    ddelta_dth = np.column_stack([g0 * du_dth[:, 0], g1 * du_dth[:, 1]])

    dC_ds1 = const * np.sum(np.sum(dF_ddelta * ddelta_ds1, axis=1))
    dC_ds2 = const * np.sum(np.sum(dF_ddelta * ddelta_ds2, axis=1))
    dC_dth = const * np.sum(np.sum(dF_ddelta * ddelta_dth, axis=1))

    return float(C), np.array([dC_ds1, dC_ds2, dC_dth], dtype=float)


def hex_lattice(a: float = 1.0) -> np.ndarray:
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, np.sqrt(3.0) * 0.5 * a])
    return np.column_stack([a1, a2]).astype(float)
