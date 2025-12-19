import numpy as np
import matplotlib.pyplot as plt
from latmatcher.core import hex_lattice, make_index_set, rotation_matrix, strain_matrix
from latmatcher.optim import maximise_coverage


def lattice_points_xy(Lxy: np.ndarray, nmax: int = 8) -> np.ndarray:
    idx = make_index_set(2*nmax + 1, 2*nmax + 1) - np.array([nmax, nmax])
    return (Lxy @ idx.T).T


def transformed_points_xy(Bxy: np.ndarray, params, nmax: int = 8) -> np.ndarray:
    s1, s2, theta = params
    S = strain_matrix(s1, s2)
    RT = rotation_matrix(theta).T
    return lattice_points_xy(S @ Bxy @ RT, nmax=nmax)


def main():
    A = hex_lattice(1.0)
    B = hex_lattice(1.015)
    m_set = make_index_set(9, 9)
    deg = np.pi/180.0
    bounds = [(-0.03, 0.03), (-0.03, 0.03), (-3.0*deg, 3.0*deg)]

    xopt, Copt, info = maximise_coverage(A, B, m_set, bounds=bounds, x0=np.array([0.0, 0.0, 0.0]))

    host = lattice_points_xy(A, 8)
    tgt0 = lattice_points_xy(B, 8)
    tgt1 = transformed_points_xy(B, xopt, 8)

    tol = 0.10
    matched = np.array([p for p in tgt1 if np.min(np.linalg.norm(host - p, axis=1)) < tol])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].scatter(host[:,0], host[:,1], s=10, label="Host (A)")
    ax[0].scatter(tgt0[:,0], tgt0[:,1], s=10, label="Target (B)")
    ax[0].set_title("Before optimisation")
    ax[0].set_aspect("equal", "box")
    ax[0].legend(frameon=False)

    ax[1].scatter(host[:,0], host[:,1], s=10, label="Host (A)")
    ax[1].scatter(tgt1[:,0], tgt1[:,1], s=10, label="Target (opt)")

    if len(matched):
        ax[1].scatter(matched[:,0], matched[:,1], s=90, facecolors="none", edgecolors="orange",
                      linewidths=2, label=f"Matched (<{tol} Å)")
        for p in matched:
            ax[1].arrow(0.0, 0.0, p[0], p[1], color="orange", width=0.004, alpha=0.35,
                        length_includes_head=True)

    ax[1].set_title("After optimisation")
    ax[1].set_aspect("equal", "box")
    ax[1].legend(frameon=False)

    for a in ax:
        a.set_xlabel("x"); a.set_ylabel("y")

    print("Optimiser:", info["success"], "|", info["message"])
    print(f"Best params: s1={xopt[0]:+.6f}, s2={xopt[1]:+.6f}, theta={xopt[2]/deg:+.6f} deg")
    plt.show()


if __name__ == "__main__":
    main()
