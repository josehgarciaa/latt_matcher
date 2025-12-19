# latmatcher.cli
from __future__ import annotations
import argparse
import numpy as np
from .cif import read_cif_atoms, lattice_xy_from_atoms, write_cif_atoms
from .core import make_index_set
from .optim import maximise_coverage
from .bilayer import build_bilayer_from_cifs


def _deg2rad(x: float) -> float:
    return float(x) * np.pi / 180.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="latmatcher")
    p.add_argument("cif_A")
    p.add_argument("cif_B")
    p.add_argument("--out", default="latmatcher_supercell.cif")
    p.add_argument("--N1", type=int, default=9)
    p.add_argument("--N2", type=int, default=9)
    p.add_argument("--sigma", type=float, default=0.12)
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--P", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--smax", type=float, default=0.03)
    p.add_argument("--theta-deg", type=float, default=3.0)
    p.add_argument("--dz", type=float, default=10.0)
    p.add_argument("--vac", type=float, default=10.0)
    a = p.parse_args(argv)

    A_atoms = read_cif_atoms(a.cif_A)
    B_atoms = read_cif_atoms(a.cif_B)
    A = lattice_xy_from_atoms(A_atoms)
    B = lattice_xy_from_atoms(B_atoms)

    m_set = make_index_set(a.N1, a.N2)
    thmax = _deg2rad(a.theta_deg)
    smax = float(a.smax)
    bounds = [(-smax, smax), (-smax, smax), (-thmax, thmax)]

    xopt, Copt, info = maximise_coverage(
        A, B, m_set,
        bounds=bounds,
        x0=np.array([0.0, 0.0, 0.0]),
        sigma=a.sigma, M=a.M, P=a.P, maxiter=a.maxiter
    )

    s1, s2, theta = float(xopt[0]), float(xopt[1]), float(xopt[2])
    combined = build_bilayer_from_cifs(a.cif_A, a.cif_B, (s1, s2, theta),
                                       interlayer_distance_A=a.dz, vacuum_extra_A=a.vac)
    write_cif_atoms(combined, a.out)

    print("latmatcher result")
    print(f"  success: {info['success']} | {info['message']}")
    print(f"  best params: s1={s1:+.6f}, s2={s2:+.6f}, theta={theta:+.6f} rad ({theta*180/np.pi:+.6f} deg)")
    print(f"  best coverage: C={Copt:.6e}")
    print(f"  wrote: {a.out}")
    return 0
