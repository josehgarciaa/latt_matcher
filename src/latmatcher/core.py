# latmatcher/core.py
# Core maths for 2D lattice matching (coverage + gradients) using filtered Fourier regularisation

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def rotation_matrix(theta: float) -> FloatArray:
    """2D rotation matrix R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def d_rotation_T_dtheta(theta: float) -> FloatArray:
    """Derivative of R(theta)^T with respect to theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, c], [-c, -s]], dtype=float)


def strain_matrix(s1: float, s2: float) -> FloatArray:
    """Diagonal in-plane strain matrix diag(1+s1, 1+s2)."""
    return np.array([[1.0 + s1, 0.0], [0.0, 1.0 + s2]], dtype=float)


def make_index_set(n1: int, n2: int) -> IntArray:
    """Dense rectangular index set {(m1,m2)} with 0<=m1<n1 and 0<=m2<n2."""
    m1 = np.arange(n1, dtype=int)
    m2 = np.arange(n2, dtype=int)
    grid = np.stack(np.meshgrid(m1, m2, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def make_integer_grid(p: int) -> IntArray:
    """Integer shifts in [-p,p]^2 used in the periodic Gaussian sum."""
    r = np.arange(-p, p + 1, dtype=int)
    grid = np.stack(np.meshgrid(r, r, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def frac_fourier_filtered(x: FloatArray, M: int, *, alpha: float = 10.0, p: int = 4) -> FloatArray:
    """
    Filtered truncated Fourier approximation of frac(x) in [0,1).

    frac(x) = 1/2 - (1/pi) sum_{k>=1} (1/k) sin(2πkx)

    We apply an exponential spectral filter:
        w_k = exp(-alpha * (k/M)^p)

    Parameters
    ----------
    x
        1D array of real values.
    M
        Truncation order (positive integer).
    alpha, p
        Filter parameters. Larger values damp high frequencies more strongly.

    Returns
    -------
    ndarray
        Approximation to frac(x) with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(1, M + 1, dtype=float)[:, None]
    w = np.exp(-alpha * (k / float(M)) ** p)
    return 0.5 - (1.0 / np.pi) * np.sum(w * (1.0 / k) * np.sin(2.0 * np.pi * k * x[None, :]), axis=0)


def d_frac_fourier_filtered_dx(x: FloatArray, M: int, *, alpha: float = 10.0, p: int = 4) -> FloatArray:
    """
    Derivative of the filtered truncated Fourier approximation to frac(x).

    d/dx frac_M(x) = -2 sum_{k=1}^M w_k cos(2π k x)

    Parameters
    ----------
    x
        1D array of real values.
    M
        Truncation order.
    alpha, p
        Filter parameters.

    Returns
    -------
    ndarray
        Derivative with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(1, M + 1, dtype=float)[:, None]
    w = np.exp(-alpha * (k / float(M)) ** p)
    return -2.0 * np.sum(w * np.cos(2.0 * np.pi * k * x[None, :]), axis=0)


def F_periodic_gaussian(delta: FloatArray, sigma: float, p_grid: IntArray) -> FloatArray:
    """
    Periodic Gaussian sum F(delta) = sum_{p in p_grid} exp(-||p-delta||^2 / (4σ^2)).

    Parameters
    ----------
    delta
        Array of displacements, shape (K,2), typically in [0,1)^2.
    sigma
        Gaussian width.
    p_grid
        Integer shifts, shape (J,2).

    Returns
    -------
    ndarray, shape (K,)
        Periodic Gaussian values.
    """
    delta = np.asarray(delta, dtype=float)
    diff = p_grid[None, :, :] - delta[:, None, :]
    r2 = np.sum(diff * diff, axis=-1)
    return np.sum(np.exp(-r2 / (4.0 * sigma * sigma)), axis=1)


def grad_F_wrt_delta(delta: FloatArray, sigma: float, p_grid: IntArray) -> FloatArray:
    """
    Gradient of the periodic Gaussian sum with respect to delta.

    Returns dF/d(delta) for each row, shape (K,2).
    """
    delta = np.asarray(delta, dtype=float)
    diff = p_grid[None, :, :] - delta[:, None, :]
    r2 = np.sum(diff * diff, axis=-1)
    w = np.exp(-r2 / (4.0 * sigma * sigma))
    return np.sum(w[:, :, None] * diff, axis=1) / (2.0 * sigma * sigma)


def coverage_and_grad(
    params: FloatArray,
    A: FloatArray,
    B: FloatArray,
    m_set: IntArray,
    *,
    sigma: float = 0.12,
    M: int = 12,
    P: int = 4,
    fourier_alpha: float = 10.0,
    fourier_p: int = 4,
) -> tuple[float, FloatArray]:
    """
    Compute the coverage objective C and its gradient w.r.t. (s1, s2, theta).

    Let params = (s1, s2, theta). Define the transformation that maps lattice points
    of B into fractional coordinates of A:

        u(m) = A^{-1} S(s1,s2) B R(theta)^T m

    Because the exact fractional-part map frac(u) = u - floor(u) is not differentiable,
    we use a filtered truncated Fourier approximation of frac() applied component-wise.

    The coverage is defined as:
        C = (2π σ^2) * Σ_m F(delta(m))

    where:
        delta(m) ≈ frac_fourier_filtered(u(m), M)
        F is a periodic Gaussian sum over integer shifts p in [-P,P]^2.

    Parameters
    ----------
    params
        Array-like of length 3: (s1, s2, theta).
    A, B
        Lattice matrices of shape (2,2) with lattice vectors as columns.
    m_set
        Integer index set, shape (K,2).
    sigma
        Gaussian width parameter.
    M
        Truncation order for the Fourier approximation.
    P
        Periodic image range for the Gaussian sum.
    fourier_alpha, fourier_p
        Spectral filter parameters.

    Returns
    -------
    C
        Coverage value.
    grad
        Gradient array of shape (3,) with respect to (s1, s2, theta).
    """
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != 3:
        raise ValueError(f"`params` must have length 3; got {params.size}.")

    s1, s2, theta = float(params[0]), float(params[1]), float(params[2])

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m_set = np.asarray(m_set, dtype=int)

    if A.shape != (2, 2) or B.shape != (2, 2):
        raise ValueError(f"`A` and `B` must have shape (2,2); got A={A.shape}, B={B.shape}.")
    if m_set.ndim != 2 or m_set.shape[1] != 2:
        raise ValueError(f"`m_set` must have shape (K,2); got {m_set.shape}.")
    if sigma <= 0:
        raise ValueError("`sigma` must be positive.")
    if M <= 0:
        raise ValueError("`M` must be positive.")
    if P <= 0:
        raise ValueError("`P` must be positive.")

    Ainv = np.linalg.inv(A)
    S = strain_matrix(s1, s2)
    RT = rotation_matrix(theta).T

    T = Ainv @ S @ B @ RT
    u = (T @ m_set.T).T  # (K,2)

    # Regularised fractional part in [0,1)
    delta = np.empty_like(u)
    delta[:, 0] = frac_fourier_filtered(u[:, 0], M, alpha=fourier_alpha, p=fourier_p)
    delta[:, 1] = frac_fourier_filtered(u[:, 1], M, alpha=fourier_alpha, p=fourier_p)

    # Coverage
    p_grid = make_integer_grid(P)
    Fv = F_periodic_gaussian(delta, sigma, p_grid)
    const = 2.0 * np.pi * sigma * sigma
    C = const * np.sum(Fv)

    # Gradient wrt delta
    dF_ddelta = grad_F_wrt_delta(delta, sigma, p_grid)

    # d(delta)/d(u) component-wise
    g0 = d_frac_fourier_filtered_dx(u[:, 0], M, alpha=fourier_alpha, p=fourier_p)
    g1 = d_frac_fourier_filtered_dx(u[:, 1], M, alpha=fourier_alpha, p=fourier_p)

    # Chain rule from u to parameters
    dS1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    dS2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)

    du_ds1 = (Ainv @ dS1 @ B @ RT @ m_set.T).T
    du_ds2 = (Ainv @ dS2 @ B @ RT @ m_set.T).T
    dRT = d_rotation_T_dtheta(theta)
    du_dtheta = (Ainv @ S @ B @ dRT @ m_set.T).T

    ddelta_ds1 = np.column_stack((g0 * du_ds1[:, 0], g1 * du_ds1[:, 1]))
    ddelta_ds2 = np.column_stack((g0 * du_ds2[:, 0], g1 * du_ds2[:, 1]))
    ddelta_dtheta = np.column_stack((g0 * du_dtheta[:, 0], g1 * du_dtheta[:, 1]))

    dC_ds1 = const * np.sum(np.sum(dF_ddelta * ddelta_ds1, axis=1))
    dC_ds2 = const * np.sum(np.sum(dF_ddelta * ddelta_ds2, axis=1))
    dC_dtheta = const * np.sum(np.sum(dF_ddelta * ddelta_dtheta, axis=1))

    grad = np.array([dC_ds1, dC_ds2, dC_dtheta], dtype=float)
    return float(C), grad


def hex_lattice(a: float = 1.0) -> FloatArray:
    """2D hexagonal lattice matrix with columns a1=(a,0), a2=(a/2, sqrt(3)a/2)."""
    a1 = np.array([a, 0.0], dtype=float)
    a2 = np.array([0.5 * a, np.sqrt(3.0) * 0.5 * a], dtype=float)
    return np.column_stack((a1, a2))
