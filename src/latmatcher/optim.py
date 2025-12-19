# latmatcher.optim
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .core import coverage_and_grad


def maximise_coverage(A: np.ndarray, B: np.ndarray, m_set: np.ndarray, bounds, x0=None,
                      sigma: float = 0.12, M: int = 12, P: int = 4, maxiter: int = 400):
    if x0 is None:
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)

    def fun(x):
        C, g = coverage_and_grad(x, A, B, m_set, sigma=sigma, M=M, P=P)
        return -C, -g

    res = minimize(lambda x: fun(x)[0], x0=x0, jac=lambda x: fun(x)[1],
                   bounds=bounds, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-10))

    best_x = np.array(res.x, dtype=float)
    best_C, _ = coverage_and_grad(best_x, A, B, m_set, sigma=sigma, M=M, P=P)
    info = {"success": bool(res.success), "message": str(res.message), "nfev": int(res.nfev)}
    return best_x, float(best_C), info
