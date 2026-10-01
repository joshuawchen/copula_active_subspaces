"""Reference subspace KL in the rank-Gaussianized coordinates (Part II):
c_r^U(u) = E_{w ~ gamma}[ c^Z(V_r u + V_perp w) ] by tensor Gauss-Hermite
quadrature and Delta_sub = E[ log c^Z(Z) - log c_r^U(U) ].
Writes cache/delta_sub_Z_{benchmark}.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from cas import analytic_oracle as ao          # noqa: E402
from cas.mixing import B_SUPPORT               # noqa: E402

R_DEFAULT = 4
_SEED = 20260806


def _grids(spec, seed):
    """Per mixed coordinate: (grid, f, F, dlogf) for the exact marginal."""
    Zmean, v = ao._marginal_mixtures(spec, np.random.default_rng(seed + 1))
    out = {}
    for k in range(B_SUPPORT):
        mk = Zmean[:, k]
        sk = float(np.sqrt(v[k]))
        g = np.linspace(mk.min() - 8.0 * sk, mk.max() + 8.0 * sk, ao._MAR_GRID)
        out[k] = (g,) + ao._grid_functions(mk, v[k], g)
    return out


def _log_c_Z(Zmix, grids, spec):
    """log c^Z at rank-Gaussianized points Zmix of shape (n, B_SUPPORT).

    Inverts the rank transform coordinatewise, then evaluates
    log pi_z(z) - sum_k log f_k(z_k) on the mixed block.
    """
    n = Zmix.shape[0]
    B = B_SUPPORT
    z = np.empty((n, B))
    log_f = np.zeros(n)
    P = stats.norm.cdf(Zmix)
    for k in range(B):
        g, f, F, _ = grids[k]
        # F is increasing on the grid; invert by interpolating grid against F.
        z[:, k] = np.interp(np.clip(P[:, k], F[0], F[-1]), F, g)
        log_f += np.log(np.maximum(np.interp(z[:, k], g, f), 1e-300))

    # log pi_z on the mixed block: z = L W restricted there, |det| = 1.
    L8 = spec.L[:B, :B]
    W = z @ L8                                   # W = L^T z, rows
    k_act = spec.k_act
    from scipy.stats import norm as _norm
    log_pi_W = np.zeros(n)
    log_pi_W += _log_pi_W_active(spec, W[:, :k_act])
    log_pi_W += (-0.5 * (W[:, k_act:B] ** 2).sum(axis=1)
                 - 0.5 * (B - k_act) * np.log(2.0 * np.pi))
    return log_pi_W - log_f


def _log_pi_W_active(spec, Wact):
    """Closed-form active-block log density for the example."""
    name = spec.name
    if name == "banana":
        from cas.noise import log_pi_W_active as f
    elif name == "even_fold":
        from cas.even_fold import log_pi_W_active as f
    elif name == "conformal_cube":
        from cas.conformal_cube import log_pi_W_active as f
    else:
        raise KeyError(name)
    return np.asarray(f(Wact), dtype=float)


def _inner_nodes(Sigma, nq, n_keep, nq_small=3, tol=1e-6):
    """Gauss-Hermite nodes for xi ~ N(0, Sigma) on the mixed block.

    Sigma = I_B - A A^T for A the mixed-block rows of the basis. At the exact
    basis A^T A = I exactly, so Sigma has B - r unit eigenvalues and r zeros
    and the product grid runs over B - r axes. At an estimated basis the r
    zeros become the squared leakage onto the padding coordinates: small at
    large N, but reaching 0.4 at N = 500, where discarding those directions
    is an error rather than an approximation.

    The node count is therefore graded by eigenvalue. The B - r leading
    directions take nq nodes each; a trailing direction takes nq_small nodes
    when its eigenvalue exceeds tol, and is evaluated at its mean otherwise.
    A three-point rule is exact through degree five, which suffices for a
    direction whose standard deviation is below one.
    """
    lam, Q = np.linalg.eigh(Sigma)
    order = np.argsort(-lam)
    lam = np.maximum(lam[order], 0.0)
    Q = Q[:, order]
    counts = np.where(np.arange(len(lam)) < n_keep, nq,
                      np.where(lam > tol, nq_small, 1))
    active = counts > 1
    lam_k, Q_k, cnt_k = lam[active], Q[:, active], counts[active]
    dropped = float(lam[~active].sum())

    axes, wts_1d = [], []
    for c in cnt_k:
        x, w = np.polynomial.hermite_e.hermegauss(int(c))
        axes.append(x)
        wts_1d.append(w / np.sqrt(2.0 * np.pi))
    mesh = np.meshgrid(*axes, indexing="ij")
    nodes = np.stack([m.ravel() for m in mesh], axis=1)
    wts = np.ones(len(nodes))
    for j, (x, w) in enumerate(zip(axes, wts_1d)):
        wts *= w[np.searchsorted(x, nodes[:, j])]
    base = (nodes * np.sqrt(lam_k)[None, :]) @ Q_k.T
    return base, wts, int(active.sum()), dropped


def delta_sub_at_basis(V_mixed, Zmix, logc_Z, grids, spec, nq=7, B_pad=None):
    """Delta_sub at an arbitrary basis, from a fixed outer sample.

    V_mixed is the mixed-block part of the basis, shape (B, r), not
    orthonormalized: its Gram deficit is the leakage onto the padding
    coordinates. B_pad, the padding part, contributes to u through an
    independent Gaussian, since the padding coordinates are standard normal and
    independent of the mixed block.
    """
    B, r = V_mixed.shape
    Sigma = np.eye(B) - V_mixed @ V_mixed.T
    base, wts, dm, dropped = _inner_nodes(Sigma, nq, B - r)
    U = Zmix @ V_mixed
    if B_pad is not None and np.abs(B_pad).max() > 0:
        # u = A^T Z_m + B^T Z_p with Z_p ~ gamma independent, so the padding
        # part enters u as N(0, B^T B) and must be drawn, not omitted.
        Cov = B_pad.T @ B_pad
        L = np.linalg.cholesky(Cov + 1e-14 * np.eye(r))
        U = U + np.random.default_rng(12345).standard_normal((len(U), r)) @ L.T
    n = len(U)
    inner_kl = np.empty(n)
    chunk = max(1, 2_000_000 // max(len(base), 1))
    for i0 in range(0, n, chunk):
        Ub = U[i0:i0 + chunk]
        pts = (Ub @ V_mixed.T)[:, None, :] + base[None, :, :]
        lc = _log_c_Z(pts.reshape(-1, B), grids, spec).reshape(len(Ub), -1)
        mx = lc.max(axis=1, keepdims=True)
        lcU = mx[:, 0] + np.log((wts[None, :] * np.exp(lc - mx)).sum(axis=1))
        lr = lc - lcU[:, None]
        inner_kl[i0:i0 + chunk] = (wts[None, :] * np.exp(lr) * lr).sum(axis=1)
    return {
        "delta_sub_Z_mean": float(inner_kl.mean()),
        "delta_sub_Z_se": float(inner_kl.std(ddof=1) / np.sqrt(n)),
        "inner_dim": dm, "dropped_variance": dropped,
    }


def run(benchmark: str, r: int = R_DEFAULT, n_outer: int = 4000,
        nq: int = 11, seed: int = _SEED) -> dict:
    spec = ao._spec(benchmark)
    B = B_SUPPORT
    grids = _grids(spec, seed)

    # Exact V_r^C, restricted to the mixed block (verify the padding rows vanish).
    path = ao._cache_path(benchmark, 8, 1_000_000, 20260715)
    V_full = np.asarray(np.load(path)["V"], dtype=float)[:, :r]
    pad = float(np.abs(V_full[B:, :]).max())
    V = V_full[:B, :]
    V, _ = np.linalg.qr(V)
    Q, _ = np.linalg.qr(np.eye(B) - V @ V.T)
    Vp = Q[:, :B - r]

    # Outer sample: Z ~ pi_Z on the mixed block.
    rng = np.random.default_rng(seed)
    W = spec.sample_W(n_outer, rng)
    zc = (W @ spec.L.T)[:, :B]
    Zmix = np.empty_like(zc)
    for k in range(B):
        g, f, F, _ = grids[k]
        Zmix[:, k] = stats.norm.ppf(
            np.clip(np.interp(zc[:, k], g, F), 1e-12, 1 - 1e-12))

    logc_Z = _log_c_Z(Zmix, grids, spec)
    U = Zmix @ V

    # Inner: c_r^U(u) = E_{w ~ gamma_{B-r}} [ c^Z(V u + Vp w) ] by tensor
    # Gauss-Hermite (probabilists' scaling).
    x, wt = np.polynomial.hermite_e.hermegauss(nq)
    wt = wt / np.sqrt(2.0 * np.pi)
    dm = B - r
    mesh = np.meshgrid(*([x] * dm), indexing="ij")
    Wnodes = np.stack([m.ravel() for m in mesh], axis=1)          # (nq^dm, dm)
    Wwts = np.ones(len(Wnodes))
    for j in range(dm):
        Wwts *= wt[np.searchsorted(x, Wnodes[:, j])]
    base = Wnodes @ Vp.T                                          # (nq^dm, B)

    log_cU = np.empty(n_outer)
    inner_kl = np.empty(n_outer)
    chunk = max(1, 2_000_000 // len(Wnodes))
    for i0 in range(0, n_outer, chunk):
        Ub = U[i0:i0 + chunk]
        pts = (Ub @ V.T)[:, None, :] + base[None, :, :]
        flat = pts.reshape(-1, B)
        lc = _log_c_Z(flat, grids, spec).reshape(len(Ub), len(Wnodes))
        mx = lc.max(axis=1, keepdims=True)
        lcU = mx[:, 0] + np.log((Wwts[None, :] * np.exp(lc - mx)).sum(axis=1))
        log_cU[i0:i0 + chunk] = lcU
        # D_KL(pi_{W|U=u} || gamma) = E_{w~gamma}[ (c/cU) log(c/cU) ], the same
        # nodes and weights, nonnegative for every u.
        lr = lc - lcU[:, None]
        inner_kl[i0:i0 + chunk] = (
            Wwts[None, :] * np.exp(lr) * lr).sum(axis=1)

    delta = inner_kl
    diff_est = logc_Z - log_cU
    out = {
        "benchmark": benchmark, "r": r, "B_SUPPORT": B,
        "n_outer": n_outer, "nq": nq, "seed": seed,
        "V_padding_max_abs": pad,
        "delta_sub_Z_mean": float(delta.mean()),
        "delta_sub_Z_se": float(delta.std(ddof=1) / np.sqrt(n_outer)),
        "delta_sub_Z_median": float(np.median(delta)),
        "delta_sub_Z_min": float(delta.min()),
        "delta_sub_Z_max": float(delta.max()),
        "diff_estimator_mean": float(diff_est.mean()),
        "diff_estimator_se": float(diff_est.std(ddof=1) / np.sqrt(n_outer)),
    }
    p = os.path.join(_REPO, "cache", f"delta_sub_Z_{benchmark}.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, p)

    print(f"\n=== {benchmark}: Delta_sub at V_r^C, rank-Gaussianized ===")
    print(f"  padding support of V_r^C: {pad:.2e} (expect ~0)")
    print(f"  Delta_sub(Z) = {out['delta_sub_Z_mean']:.5f} "
          f"+/- {out['delta_sub_Z_se']:.5f}   (inner-KL estimator)")
    print(f"  difference estimator: {out['diff_estimator_mean']:+.5f} "
          f"+/- {out['diff_estimator_se']:.5f}")
    print(f"  saved {p}")
    return out


def run_cells(benchmark: str, r: int = R_DEFAULT, n_outer: int = 3000,
              nq: int = 7, seed: int = _SEED) -> dict:
    """Delta_sub at every estimated basis of the cor44 scan, by tensor
    Gauss-Hermite quadrature (nq nodes per axis) in the rank-Gaussianized
    coordinates.
    """
    spec = ao._spec(benchmark)
    B = B_SUPPORT
    grids = _grids(spec, seed)

    rng = np.random.default_rng(seed)
    W = spec.sample_W(n_outer, rng)
    zc = (W @ spec.L.T)[:, :B]
    Zmix = np.empty_like(zc)
    for k in range(B):
        g, f, F, _ = grids[k]
        Zmix[:, k] = stats.norm.ppf(
            np.clip(np.interp(zc[:, k], g, F), 1e-12, 1 - 1e-12))
    logc_Z = _log_c_Z(Zmix, grids, spec)

    scan = json.load(open(os.path.join(
        _REPO, "cache", f"cor44_{benchmark}_scan_K4_q2_r{r}.json")))

    cells = []
    for c in scan:
        V_full = np.asarray(c["V_hat"], dtype=float)
        res = delta_sub_at_basis(V_full[:B, :], Zmix, logc_Z, grids, spec,
                                 nq=nq, B_pad=V_full[B:, :])
        res.update({"N": int(c["N"]), "seed": int(c["seed"]),
                    "padding_leak": float(np.abs(V_full[B:, :]).max())})
        cells.append(res)
        print(f"  N={res['N']:>7} seed={res['seed']:>2}  "
              f"Delta_sub={res['delta_sub_Z_mean']:.5f} "
              f"+/- {res['delta_sub_Z_se']:.5f}  "
              f"(inner dim {res['inner_dim']}, dropped var "
              f"{res['dropped_variance']:.2e}, leak "
              f"{res['padding_leak']:.3f})", flush=True)

    out = {"benchmark": benchmark, "r": r, "n_outer": n_outer, "nq": nq,
           "seed": seed, "cells": cells}
    p = os.path.join(_REPO, "cache", f"delta_sub_Z_cells_{benchmark}.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, p)
    print(f"\n  {'N':>7} {'mean Delta_sub':>15} {'sd across seeds':>16}")
    for N in sorted({c["N"] for c in cells}):
        v = np.array([c["delta_sub_Z_mean"] for c in cells if c["N"] == N])
        print(f"  {N:>7} {v.mean():>15.5f} {v.std(ddof=1):>16.5f}")
    print(f"  saved {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="all",
                    choices=["banana", "even_fold", "conformal_cube", "all"])
    ap.add_argument("--n_outer", type=int, default=4000)
    ap.add_argument("--nq", type=int, default=11)
    ap.add_argument("--cells", action="store_true",
                    help="evaluate at every estimated basis of the scan "
                         "instead of at the exact V_r^C")
    args = ap.parse_args()
    for b in (["banana", "even_fold", "conformal_cube"]
              if args.benchmark == "all" else [args.benchmark]):
        if args.cells:
            print(f"\n=== {b}: Delta_sub at the estimated bases ===")
            run_cells(b, n_outer=args.n_outer, nq=args.nq)
        else:
            run(b, n_outer=args.n_outer, nq=args.nq)


if __name__ == "__main__":
    main()
