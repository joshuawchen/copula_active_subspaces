"""Reference subspace V_r^C for Examples 1-3.

Computes C = E[S S^T] from the analytic copula score, using the Gaussian-mixture
marginals of the mixed latent coordinates, and returns its top-r eigenspace.
Coordinates outside the mixing support have zero score, so the rank of C is at
most B_SUPPORT.
"""
from __future__ import annotations
import dataclasses
import os
import numpy as np
from scipy import stats

from .mixing import B_SUPPORT

_MAR_N = 200_000          # MC draws for the marginal mixtures
_MAR_GRID = 4001          # grid points per mixed coordinate
_REF_N = 1_000_000        # MC draws for C
_REF_SEED = 20260715
_R_MAX = 8


@dataclasses.dataclass
class LawSpec:
    name: str
    L: np.ndarray            # (D, D) orthogonal mixing
    D: int
    k_act: int               # active block size (cols 0..k_act-1)
    input_cols: list         # Gaussian block inputs
    output_cols: list        # conditionally-Gaussian block outputs (order matches cond_means)
    s2: float                # conditional variance of the outputs
    cond_means: object       # (Win: (n, n_in)) -> (n, n_out) output conditional means
    grad_active: object      # grad_log_pi_W_active: (n, k_act) -> (n, k_act)
    sample_W: object         # (n, rng) -> (n, D) latent W


def _spec(tag):
    if tag == "banana":
        from . import noise as m
        return LawSpec(
            "banana", m.L_CHOL, m.D_OBS, 2 * m.K_BAN, [0, 1], [2, 3], m._LD_S2,
            lambda Win: np.column_stack(m._conf_means(Win[:, 0], Win[:, 1])),
            m.grad_log_pi_W_active, m.sample_banana_W)
    if tag == "even_fold":
        from . import even_fold as m
        return LawSpec(
            "even_fold", m.L_CHOL_EF, m.D_EF, 2 * m.N_PAIRS_EF, [0, 2], [1, 3], m._EF_S2,
            lambda Win: np.column_stack([m._ef_mu(Win[:, 0]), m._ef_mu(Win[:, 1])]),
            m.grad_log_pi_W_active, m.sample_even_fold_W)
    if tag == "conformal_cube":
        from . import conformal_cube as m
        return LawSpec(
            "conformal_cube", m.L_CHOL_C3, m.D_C3, 4, [0, 1], [2, 3], m._C3_S2,
            lambda Win: np.column_stack(m._c3_means(Win[:, 0], Win[:, 1])),
            m.grad_log_pi_W_active, m.sample_conformal_cube_W)
    raise KeyError(f"analytic_oracle: unknown law '{tag}'")


def _marginal_mixtures(spec, rng):
    """Per mixed coord k in range(B_SUPPORT): conditional means m_k (shape
    (_MAR_N,)) and variance v_k defining f_{z_k}(t) = mean_m N(t; m_k[m], v_k)."""
    L = spec.L
    W = spec.sample_W(_MAR_N, rng)
    Win = W[:, spec.input_cols]
    Mout = spec.cond_means(Win)
    Wmean = np.zeros((_MAR_N, spec.D))
    Wmean[:, spec.input_cols] = Win
    Wmean[:, spec.output_cols] = Mout
    Zmean = Wmean @ L.T                       # column k holds m_k^{(m)}
    var_col = np.ones(spec.D)
    var_col[spec.input_cols] = 0.0            # inputs are conditioned on
    var_col[spec.output_cols] = spec.s2       # outputs: conditional variance s2
    v = (L ** 2) @ var_col                    # (D,): v_k = Var(z_k | inputs)
    return Zmean, v


def _grid_functions(m_samp, v, grid):
    """f(grid), F(grid), and dlogf(grid) = (log f)'(grid) for the Gaussian
    mixture {N(m_samp[m], v)}_m, evaluated by chunking over components."""
    s = float(np.sqrt(v)); n = m_samp.shape[0]; G = grid.shape[0]
    inv = 1.0 / v
    f = np.zeros(G); fp = np.zeros(G); F = np.zeros(G)
    step = max(1, 4_000_000 // G)
    root = 1.0 / (s * np.sqrt(2.0 * np.pi))
    for i0 in range(0, n, step):
        mm = m_samp[i0:i0 + step]
        d = grid[:, None] - mm[None, :]
        pdf = np.exp(-0.5 * (d * d) * inv) * root
        f += pdf.sum(1)
        fp += (-d * inv * pdf).sum(1)
        F += stats.norm.cdf(d / s).sum(1)
    f /= n; fp /= n; F /= n
    dlogf = fp / np.maximum(f, 1e-300)
    return f, F, dlogf


def _exact_score(spec, n_ref=_REF_N, seed=_REF_SEED):
    """Exact rank-Gaussianized coordinates and copula score.

    Returns (Z, S) of shape (n_ref, D): Z is the rank transform of the mixed
    latent z = L W, exactly (in the MC limit) rather than through a logistic
    approximation, and S is the exact copula score at those coordinates. The
    operations and the RNG draws are in the order compute_C has always used, so
    the cached analytic_VrC_* files reproduce bit for bit."""
    B = B_SUPPORT; D = spec.D
    Zmean, v = _marginal_mixtures(spec, np.random.default_rng(seed + 1))
    grids = {}
    for k in range(B):
        mk = Zmean[:, k]; sk = float(np.sqrt(v[k]))
        grid = np.linspace(mk.min() - 8.0 * sk, mk.max() + 8.0 * sk, _MAR_GRID)
        grids[k] = (grid,) + _grid_functions(mk, v[k], grid)

    rng = np.random.default_rng(seed)
    W = spec.sample_W(n_ref, rng)
    z = W @ spec.L.T
    gW = np.empty((n_ref, D))
    gW[:, :spec.k_act] = spec.grad_active(W[:, :spec.k_act])
    gW[:, spec.k_act:] = -W[:, spec.k_act:]
    a = gW @ spec.L.T                          # (L grad_W log pi_W)_k in column k

    Z = np.empty((n_ref, D))
    S = np.zeros((n_ref, D))
    for k in range(D):
        zk = z[:, k]
        if k < B:
            grid, f, F, dlogf = grids[k]
            fk = np.interp(zk, grid, f)
            Fk = np.clip(np.interp(zk, grid, F), 1e-12, 1.0 - 1e-12)
            dlk = np.interp(zk, grid, dlogf)
            Zk = stats.norm.ppf(Fk)
            Z[:, k] = Zk
            S[:, k] = (stats.norm.pdf(Zk) / np.maximum(fk, 1e-300)) * (a[:, k] - dlk)
        else:
            Z[:, k] = zk                       # standard-Gaussian marginal
            S[:, k] = a[:, k] + zk             # identically 0
    return Z, S


def compute_C(spec, r_max=_R_MAX, n_ref=_REF_N, seed=_REF_SEED):
    """Exact copula-score covariance C and its top-r_max eigenspace.

    Returns (V (D, r_max), top eigenvalues (r_max,), tr C)."""
    _Z, S = _exact_score(spec, n_ref=n_ref, seed=seed)
    C = S.T @ S / n_ref
    w, Q = np.linalg.eigh(C)
    order = np.argsort(-w)
    return Q[:, order][:, :r_max], w[order][:r_max], float(np.trace(C))


def compute_MHC(spec, n_ref=_REF_N, seed=_REF_SEED):
    """Second moment M, density-score covariance H, and copula-score covariance
    C, all in the exact rank-Gaussianized coordinates.

    The three matrices the LCLMZ bound J_hi needs. Because the rank transform
    makes every marginal of Z standard Gaussian, tr(M) = D up to Monte Carlo
    error, and the density score is grad log pi_Z = S - Z by the definition of
    the copula score, so

        M = E[Z Z^T],   H = E[(S - Z)(S - Z)^T],   C = E[S S^T].

    Computing these from the exact copula score, rather than from the mixed
    latent z = L W, is what distinguishes them from the earlier W-space
    surrogate: under an orthogonal L that surrogate cancels the padding
    coordinates exactly, giving C of exact rank k_act and an H whose complement
    block is the identity, so both bounds collapse to law-independent constants.
    """
    Z, S = _exact_score(spec, n_ref=n_ref, seed=seed)
    G = S - Z
    return {
        "M": Z.T @ Z / n_ref,
        "H": G.T @ G / n_ref,
        "C": S.T @ S / n_ref,
        "d": spec.D,
        "z_mean": Z.mean(axis=0),
    }


def _cache_path(tag, r_max, n_ref, seed):
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    return os.path.join(d, f"analytic_VrC_{tag}_top{r_max}_Nref{n_ref}_seed{seed}.npz")


def analytic_oracle_subspace(tag, r, r_max=_R_MAX, n_ref=_REF_N, seed=_REF_SEED,
                             rebuild=False):
    """Top-r columns of the exact V_r^C for `tag`, cached to disk.

    The exact copula-score covariance is built once (compute_C) and cached; this
    is the ground-truth reference the paper's V_r^C and the subspace-recovery
    sinTheta are measured against. No (K,q) score-matching proxy."""
    if r > r_max:
        raise ValueError(f"analytic_oracle_subspace: r={r} exceeds r_max={r_max}")
    path = _cache_path(tag, r_max, n_ref, seed)
    if (not rebuild) and os.path.exists(path):
        return np.load(path)["V"][:, :r]
    V, eigs, trC = compute_C(_spec(tag), r_max=r_max, n_ref=n_ref, seed=seed)
    d = os.path.dirname(path)
    if os.path.isdir(d):
        np.savez(path, V=V, eigvals_top=eigs, trC=trC, tag=tag,
                 n_ref=n_ref, seed=seed, r_max=r_max)
    return V[:, :r]
