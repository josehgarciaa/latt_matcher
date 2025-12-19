# latmatcher/optim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from .core import coverage_and_grad

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class OptimisationInfo:
    """Compact optimiser diagnostics for `maximise_coverage`."""
    success: bool
    message: str
    nfev: int
    nit: int | None = None


def maximise_coverage(
    A: FloatArray,
    B: FloatArray,
    m_set: IntArray,
    bounds: Sequence[tuple[float, float]],
    *,
    x0: Sequence[float] | FloatArray | None = None,
    sigma: float = 0.01,
    M: int = 12,
    P: int = 4,
    maxiter: int = 400,
    ftol: float = 1e-10,
    options: Mapping[str, Any] | None = None,
) -> tuple[FloatArray, float, OptimisationInfo]:
    """
    Maximise the lattice-coverage objective using L-BFGS-B with analytic gradients.

    The optimisation variables are:

    - ``s1``: in-plane strain along x (dimensionless)
    - ``s2``: in-plane strain along y (dimensionless)
    - ``theta``: in-plane rotation angle (radians)

    Internally, we call :func:`latmatcher.core.coverage_and_grad` which returns:

    - ``C``: coverage (correlation) between A and the transformed B
    - ``g``: gradient of C with respect to (s1, s2, theta)

    Since SciPy's optimiser *minimises*, we minimise ``-C`` with gradient ``-g``.

    Parameters
    ----------
    A, B
        Arrays of shape (2, 2). Columns are the in-plane lattice vectors.
        ``A`` is the reference lattice; ``B`` is the lattice being strained/rotated.
    m_set
        Integer array of shape (K, 2) defining sampled points of lattice B.
        Each row is an index pair (m1, m2). Increasing K generally improves robustness
        but increases runtime.
    bounds
        Box bounds for ``(s1, s2, theta)`` as
        ``[(s1_min, s1_max), (s2_min, s2_max), (theta_min, theta_max)]``.
    x0
        Initial guess for ``(s1, s2, theta)``. If None, the midpoint of each bound
        interval is used.
    sigma
        Gaussian width used in the periodic correlation (in Å, consistent with lattices).
        Smaller values enforce stricter matching.
    M
        Truncation order for the Fourier approximation of the fractional-part map
        (used to avoid nondifferentiability from the floor function).
    P
        Periodic image range in the Gaussian sum; uses integer shifts in ``[-P, P]^2``.
    maxiter
        Maximum number of L-BFGS-B iterations.
    ftol
        Function tolerance passed to SciPy (via ``options``).
    options
        Extra SciPy L-BFGS-B options. Values here override defaults.

    Returns
    -------
    xopt
        Optimal parameters as a float array ``[s1, s2, theta]``.
    Copt
        Coverage evaluated at ``xopt``.
    info
        Optimiser diagnostics (success flag, message, evaluations, iterations).

    Raises
    ------
    ValueError
        If inputs have incompatible shapes, invalid bounds, or non-finite values.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m_set = np.asarray(m_set, dtype=int)

    if A.shape != (2, 2) or B.shape != (2, 2):
        raise ValueError(f"`A` and `B` must have shape (2, 2); got A={A.shape}, B={B.shape}.")
    if m_set.ndim != 2 or m_set.shape[1] != 2:
        raise ValueError(f"`m_set` must have shape (K, 2); got {m_set.shape}.")
    if len(bounds) != 3:
        raise ValueError("`bounds` must contain exactly 3 (min, max) pairs for (s1, s2, theta).")

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (3, 2):
        raise ValueError(f"`bounds` must be a sequence of 3 pairs; got array shape {bounds_arr.shape}.")
    if np.any(~np.isfinite(bounds_arr)):
        raise ValueError("`bounds` must be finite.")
    if np.any(bounds_arr[:, 0] > bounds_arr[:, 1]):
        raise ValueError("Each bound must satisfy min <= max.")

    if x0 is None:
        x0_arr = np.mean(bounds_arr, axis=1)
    else:
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.size != 3:
            raise ValueError(f"`x0` must have length 3; got {x0_arr.size}.")
    if np.any(~np.isfinite(x0_arr)):
        raise ValueError("`x0` must be finite.")

    # Ensure initial guess lies within bounds (project if necessary).
    x0_arr = np.clip(x0_arr, bounds_arr[:, 0], bounds_arr[:, 1])

    if maxiter <= 0:
        raise ValueError("`maxiter` must be positive.")
    if sigma <= 0:
        raise ValueError("`sigma` must be positive.")
    if M <= 0:
        raise ValueError("`M` must be positive.")
    if P <= 0:
        raise ValueError("`P` must be positive.")

    def objective(x: FloatArray) -> float:
        print("x=",x)
        C, _ = coverage_and_grad(x, A, B, m_set, sigma=sigma, M=M, P=P)
        return float(-C)

    def gradient(x: FloatArray) -> FloatArray:
        _, g = coverage_and_grad(x, A, B, m_set, sigma=sigma, M=M, P=P)
        return np.asarray(-g, dtype=float)

    opt: dict[str, Any] = {"maxiter": int(maxiter), "ftol": float(ftol)}
    if options:
        opt.update(dict(options))

    res: OptimizeResult = scipy_minimize(
        objective,
        x0=x0_arr,
        jac=gradient,
        bounds=[(float(lo), float(hi)) for lo, hi in bounds_arr],
        method="L-BFGS-B",
        options=opt,
    )

    xopt = np.asarray(res.x, dtype=float)
    Copt, _ = coverage_and_grad(xopt, A, B, m_set, sigma=sigma, M=M, P=P)

    info = OptimisationInfo(
        success=bool(res.success),
        message=str(res.message),
        nfev=int(getattr(res, "nfev", -1)),
        nit=int(getattr(res, "nit", None)) if getattr(res, "nit", None) is not None else None,
    )

    return xopt, float(Copt), info
