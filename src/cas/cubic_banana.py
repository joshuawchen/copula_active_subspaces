"""Cubic-banana noise law on d = 25 coordinates: three axis-aligned pairs with
W_{2k+1} = (alpha / sqrt(15)) W_{2k}^3 + sqrt(1 - alpha^2) eps_k, alpha = 0.6,
and standard-normal padding.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator

from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

D_CB: int = 25
R_CB_TARGET: int = 6
N_PAIRS: int = 3  # banana pairs at coords (0,1), (2,3), (4,5)

# Cubic perturbation strength.  At alpha = 0.6, the cubic-rescaled cube
# (alpha/sqrt(15)) * W_1^3 carries 36% of the variance of W_{2k+1}, with
# the remaining 64% from the independent Gaussian eps.  Strong enough to
# give a clear rank-6 cliff (top-3 eigvals ~1.4, next-3 eigvals ~0.4).
ALPHA_CB: float = 0.6

# Marginal-of-W_2 cache parameters.  The W_2 marginal is a 1D density
# (Gaussian convolution of a cubic-rescaled Gaussian) — same density for
# all three banana pairs by symmetry.
#
# The table is built on first use, or loaded from its cache file; importing
# the package does not build it.
_W2_GRID_LO: float = -12.0
_W2_GRID_HI: float = 12.0
_W2_GRID_N: int = 4097
_W2_MC_N: int = 10_000_000
_W2_SEED: int = 20260514


# ---------------------------------------------------------------------------
# W_2 marginal: built once at module load via Gaussian-KDE
# ---------------------------------------------------------------------------

def _build_W2_marginal_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build smooth CDF and PDF tables for the W_2 marginal.

    Identity: W_2 = (alpha/sqrt(15)) * W_1^3 + sqrt(1-alpha^2) * eps, with
    W_1 ~ N(0, 1), eps ~ N(0, 1), W_1 ⊥ eps.  Hence
        F_W2(w) = E_{W_1} [ Phi((w - mu(W_1)) / sigma_eps) ]
        f_W2(w) = E_{W_1} [ phi((w - mu(W_1)) / sigma_eps) / sigma_eps ]
    where mu(W_1) = (alpha/sqrt(15)) * W_1^3, sigma_eps = sqrt(1 - alpha^2).

    Evaluated on a fine grid via _W2_MC_N Monte Carlo samples shared
    across the CDF and PDF.  This is a Gaussian-KDE form — the resulting
    tables are smooth in w (no kinks at grid spacing).

    Cached to disk in the repository's `cache/` directory.  Cache key
    encodes (alpha, M_MC, grid_N, grid_lo, grid_hi, seed); changing any
    of these forces a rebuild on next use.
    """
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
    cache_dir = os.path.abspath(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"cubic_banana_W2_marg_a{ALPHA_CB:.3f}_M{_W2_MC_N}_"
        f"G{_W2_GRID_N}_seed{_W2_SEED}.npz",
    )
    if os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["grid"], d["cdf"], d["pdf"]

    rng = np.random.default_rng(_W2_SEED)
    W1 = rng.standard_normal(_W2_MC_N)
    mu = (ALPHA_CB / np.sqrt(15.0)) * W1 ** 3
    sigma_eps = np.sqrt(1.0 - ALPHA_CB ** 2)

    grid = np.linspace(_W2_GRID_LO, _W2_GRID_HI, _W2_GRID_N)
    cdf = np.zeros(_W2_GRID_N)
    pdf = np.zeros(_W2_GRID_N)

    # Chunked evaluation to manage memory: blocks of (chunk_m, chunk_g).
    chunk_m = 100_000
    chunk_g = 256
    for m0 in range(0, _W2_MC_N, chunk_m):
        m1 = min(m0 + chunk_m, _W2_MC_N)
        mu_c = mu[m0:m1]
        for g0 in range(0, _W2_GRID_N, chunk_g):
            g1 = min(g0 + chunk_g, _W2_GRID_N)
            diff = grid[None, g0:g1] - mu_c[:, None]
            scaled = diff / sigma_eps
            cdf[g0:g1] += stats.norm.cdf(scaled).sum(axis=0)
            pdf[g0:g1] += (stats.norm.pdf(scaled) / sigma_eps).sum(axis=0)
    cdf /= _W2_MC_N
    pdf /= _W2_MC_N
    pdf = np.maximum(pdf, 1e-300)

    if os.path.isdir(cache_dir):
        np.savez(cache_file, grid=grid, cdf=cdf, pdf=pdf)
    return grid, cdf, pdf


_W2_TABLES = None


def _w2_tables():
    """The W_2 marginal's grid and its PCHIP cdf, pdf and inverse cdf.

    Built on the first call, so that importing the package does not build
    the table: when the package is installed rather than run from the
    repository, the cache directory is absent and the build takes minutes.
    """
    global _W2_TABLES
    if _W2_TABLES is None:
        grid, cdf, pdf = _build_W2_marginal_tables()
        # Inverse CDF: dedup any flat regions (MC noise can create exact ties).
        cdf_unique, idx_unique = np.unique(cdf, return_index=True)
        _W2_TABLES = (
            grid,
            PchipInterpolator(grid, cdf, extrapolate=False),
            PchipInterpolator(grid, pdf, extrapolate=False),
            PchipInterpolator(cdf_unique, grid[idx_unique], extrapolate=False),
        )
    return _W2_TABLES


def _eval_F_W2(w: np.ndarray) -> np.ndarray:
    """CDF of the W_2 marginal at points w."""
    grid, cdf, _, _ = _w2_tables()
    w_clip = np.clip(w, grid[0], grid[-1])
    return cdf(w_clip)


def _eval_Finv_W2(u: np.ndarray) -> np.ndarray:
    """Inverse CDF of the W_2 marginal at probabilities u ∈ (0, 1)."""
    _, _, _, invcdf = _w2_tables()
    u_clip = np.clip(u, invcdf.x[0], invcdf.x[-1])
    return invcdf(u_clip)


def _eval_log_f_W2(w: np.ndarray) -> np.ndarray:
    """log PDF of the W_2 marginal at points w."""
    grid, _, pdf, _ = _w2_tables()
    w_clip = np.clip(w, grid[0], grid[-1])
    return np.log(np.maximum(pdf(w_clip), 1e-300))


# ---------------------------------------------------------------------------
# Public sampler and density
# ---------------------------------------------------------------------------

def sample_cubic_banana_noise(N: int, rng: np.random.Generator) -> np.ndarray:
    """Draw N samples from the cubic-banana noise law on R^{D_CB}.

    Returns eta of shape (N, d).

    Stable transform: uses logit_phi (no clipping) to map W -> eta.
    """
    W = rng.standard_normal((N, D_CB))
    for k in range(N_PAIRS):
        # Replace W[:, 2k+1] with the cubic-perturbed version
        W_pre = W[:, 2 * k + 1]
        W[:, 2 * k + 1] = (
            (ALPHA_CB / np.sqrt(15.0)) * W[:, 2 * k] ** 3
            + np.sqrt(1.0 - ALPHA_CB ** 2) * W_pre
        )
    # z = W (L = I), eta = logit(Phi(z)) computed stably (no clipping).
    return logit_phi(W)


# Precomputed constants (avoid re-evaluating in the hot path).
_CB_SCALE = ALPHA_CB / np.sqrt(15.0)         # cubic coefficient
_CB_INV_S2 = 1.0 / (1.0 - ALPHA_CB ** 2)     # 1 / (1 - alpha^2)
_CB_LOG_S2 = np.log(1.0 - ALPHA_CB ** 2)     # log(1 - alpha^2)
_CB_LOG_2PI = np.log(2.0 * np.pi)


def log_cubic_banana_density(eta: np.ndarray) -> np.ndarray:
    """Log-density of the cubic-banana noise at points eta ∈ R^{N × d}.

    Chain: eta → z (via z_j = Phi^{-1}(sigm(eta_j))) → W → log pi_W(W) → density.

    Density:
        log pi_eta(eta) = log pi_W(W(eta)) + sum_j log |dW_j/deta_j|

    Since z = W (L = I), the chain is just:
        eta_j = logit(Phi(W_j))   componentwise,
        W_j   = Phi^{-1}(sigm(eta_j)).

    Stable: uses inv_logit_phi (no clipping) and logit_phi_log_jacobian_z
    for the Jacobian (also no clipping).
    """
    eta = np.asarray(eta, dtype=np.float64)
    # Stable inverse: eta -> W without clipping.
    W = inv_logit_phi(eta)

    # log pi_W(W): three banana pairs + (d - 6) standard Gaussians.
    w_first = W[:, 0:2 * N_PAIRS:2]          # (N, N_PAIRS) — coords 0, 2, 4
    w_second = W[:, 1:2 * N_PAIRS:2]         # (N, N_PAIRS) — coords 1, 3, 5
    mu = _CB_SCALE * (w_first ** 3)
    resid = w_second - mu
    # Per-pair contribution, summed over the 3 pairs
    log_pi_pairs = (
        -0.5 * (w_first * w_first).sum(axis=1)
        - 0.5 * (resid * resid).sum(axis=1) * _CB_INV_S2
        - 0.5 * N_PAIRS * _CB_LOG_S2
        - N_PAIRS * _CB_LOG_2PI
    )
    # Padding coords (j >= 2*N_PAIRS): i.i.d. N(0, 1)
    W_rest = W[:, 2 * N_PAIRS:]
    n_rest = D_CB - 2 * N_PAIRS
    log_pi_rest = (
        -0.5 * (W_rest * W_rest).sum(axis=1)
        - 0.5 * n_rest * _CB_LOG_2PI
    )
    log_pi_W = log_pi_pairs + log_pi_rest

    # log Jacobian computed stably from W (= Zlat here since L = I).
    log_jac = logit_phi_log_jacobian_z(W).sum(axis=1)

    return log_pi_W + log_jac


# ---------------------------------------------------------------------------
# Oracle subspace: top-r eigenvectors of population score covariance C
# ---------------------------------------------------------------------------
#
# By construction, the rank-Gaussianized copula score is supported on
# coords 0..5 (the active pairs); coords 6..24 have zero score contribution.
# Hence C has EXACT rank 6 with support span{e_0, ..., e_5} ⊂ R^25.
# Any orthonormal basis of this 6-dim subspace is a valid V_KL.
# We return the canonical basis [e_0, ..., e_{r-1}] for r ≤ 6.

def oracle_subspace_cubic_banana(r: int) -> np.ndarray:
    """V_KL: top-r eigenspace of the population score covariance C.

    By the axis-aligned construction, C has exact rank 6 with support
    span{e_0, ..., e_5}.  For r ≤ 6, V_KL is any orthonormal basis of
    span{e_0, ..., e_{r-1}}, and we return the canonical basis.
    """
    if r > 2 * N_PAIRS:
        raise ValueError(
            f"oracle_subspace_cubic_banana: r={r} exceeds active rank "
            f"r* = {2 * N_PAIRS}"
        )
    V = np.zeros((D_CB, r))
    for j in range(r):
        V[j, j] = 1.0
    return V
