"""Reference quantities in the rank-Gaussianized coordinates Z.

    grids(spec)              marginals of the mixed latent coordinates
    log_c_Z(Zmix, ...)       log copula density at rank-Gaussianized points
    sample_Z(M, spec, ...)   draws of Z
    log_pi_U(U, V_r, ...)    log density of U = V_r^T Z, by tensor
                             Gauss-Hermite quadrature over the complement
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import analytic_oracle as ao
from .mixing import B_SUPPORT

__all__ = ["grids", "log_c_Z", "sample_Z", "sample_U", "log_pi_U",
           "inner_nodes", "spec_for"]

_DEFAULT_SEED = 20260806


def spec_for(benchmark: str):
    return ao._spec(benchmark)


_GRID_CACHE: dict = {}


def grids(spec, seed: int = _DEFAULT_SEED) -> dict:
    """Per mixed coordinate: (grid, f, F, dlogf) for the exact marginal.

    Memoized on (name, seed): the mixture build draws _MAR_N samples per
    coordinate and dominates the cost, while the experiments call this once
    per cell.
    """
    key = (spec.name, seed)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    Zmean, v = ao._marginal_mixtures(spec, np.random.default_rng(seed + 1))
    out = {}
    for k in range(B_SUPPORT):
        mk = Zmean[:, k]
        sk = float(np.sqrt(v[k]))
        g = np.linspace(mk.min() - 8.0 * sk, mk.max() + 8.0 * sk, ao._MAR_GRID)
        out[k] = (g,) + ao._grid_functions(mk, v[k], g)
    _GRID_CACHE[key] = out
    return out


def _log_pi_W_active(spec, Wact):
    name = spec.name
    if name == "banana":
        from .noise import log_pi_W_active as f
    elif name == "even_fold":
        from .even_fold import log_pi_W_active as f
    elif name == "conformal_cube":
        from .conformal_cube import log_pi_W_active as f
    else:
        raise KeyError(name)
    return np.asarray(f(Wact), dtype=float)


def log_c_Z(Zmix, gr, spec):
    """log c^Z at rank-Gaussianized points Zmix of shape (n, B_SUPPORT)."""
    n = Zmix.shape[0]
    B = B_SUPPORT
    z = np.empty((n, B))
    log_f = np.zeros(n)
    P = stats.norm.cdf(Zmix)
    for k in range(B):
        g, f, F, _ = gr[k]
        z[:, k] = np.interp(np.clip(P[:, k], F[0], F[-1]), F, g)
        log_f += np.log(np.maximum(np.interp(z[:, k], g, f), 1e-300))

    L8 = spec.L[:B, :B]
    W = z @ L8
    k_act = spec.k_act
    lp = _log_pi_W_active(spec, W[:, :k_act])
    lp = lp + (-0.5 * (W[:, k_act:B] ** 2).sum(axis=1)
               - 0.5 * (B - k_act) * np.log(2.0 * np.pi))
    return lp - log_f


def sample_Z(M: int, spec, gr, seed: int):
    """Draws of the full rank-Gaussianized Z, shape (M, D).

    The padding coordinates are standard normal and equal to their latent
    counterparts; the mixed block is transformed through its exact marginal.
    """
    B = B_SUPPORT
    rng = np.random.default_rng(seed)
    W = spec.sample_W(M, rng)
    z = W @ spec.L.T
    Z = z.copy()
    for k in range(B):
        g, f, F, _ = gr[k]
        Z[:, k] = stats.norm.ppf(
            np.clip(np.interp(z[:, k], g, F), 1e-12, 1.0 - 1e-12))
    return Z


def sample_U(M: int, V_r, spec, gr, seed: int):
    """Draws of U = V_r^T Z with Z rank-Gaussianized, shape (M, r)."""
    return sample_Z(M, spec, gr, seed) @ np.asarray(V_r, dtype=float)


def inner_nodes(Sigma, nq: int, n_keep: int, nq_small: int = 3,
                tol: float = 1e-6):
    """Graded Gauss-Hermite nodes for xi ~ N(0, Sigma) on the mixed block.

    The n_keep leading directions take nq nodes; a trailing direction takes
    nq_small nodes when its eigenvalue exceeds tol and is evaluated at its mean
    otherwise. A three-point rule is exact through degree five, which suffices
    for a direction whose standard deviation is below one. Returns
    (nodes on the mixed block, weights, number of integrated axes,
    variance evaluated at the mean).
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


def log_pi_U(U, V_r, spec, gr, nq: int = 7, chunk_nodes: int = 2_000_000):
    """log pi_U at supplied points U, for U = V_r^T Z and Z rank-Gaussianized.

    pi_U = c_r^U gamma_r, and c_r^U(u) is the Gaussian expectation of the
    copula density over the complement, so

        log pi_U(u) = log c_r^U(u) + log gamma_r(u).

    Returns (log_pi_U, log_c_r_U, diagnostics).
    """
    U = np.asarray(U, dtype=float)
    V_r = np.asarray(V_r, dtype=float)
    B = B_SUPPORT
    r = V_r.shape[1]
    A = V_r[:B, :]
    Sigma = np.eye(B) - A @ A.T
    base, wts, dim, dropped = inner_nodes(Sigma, nq, B - r)

    out = np.empty(len(U))
    chunk = max(1, chunk_nodes // max(len(base), 1))
    for i0 in range(0, len(U), chunk):
        Ub = U[i0:i0 + chunk]
        pts = (Ub @ A.T)[:, None, :] + base[None, :, :]
        lc = log_c_Z(pts.reshape(-1, B), gr, spec).reshape(len(Ub), -1)
        mx = lc.max(axis=1, keepdims=True)
        out[i0:i0 + chunk] = mx[:, 0] + np.log(
            (wts[None, :] * np.exp(lc - mx)).sum(axis=1))

    log_gamma_r = -0.5 * (U * U).sum(axis=1) - 0.5 * r * np.log(2.0 * np.pi)
    return out + log_gamma_r, out, {"inner_dim": dim,
                                    "dropped_variance": dropped}
