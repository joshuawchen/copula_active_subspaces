"""Polynomial-perturbed Gaussian noise law on d = 20 coordinates:
f(W) proportional to exp(-|W|^2 / 2 - sum_k lam_k phi(b_k^T W)), with
b_k = (e_{2k} + e_{2k+1}) / sqrt(2), phi(x) = x^4 / 12 and lam_k = 4.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

D_PPG: int = 20
R_PPG_PRIMARY: int = 3   # primary active rank (moderate cliff at this index)
R_PPG_TARGET: int = 3    # deployment rank for experiments (matches primary)
N_PAIRS: int = 3         # active pairs at coords (0,1), (2,3), (4,5)

# Quartic-coupling strengths per pair.  Equal lambdas give a clean cliff
# at the primary rank (~4.4 in eigval ratio l_3/l_4 vs the secondary ring).
LAMBDAS_PPG: tuple[float, float, float] = (4.0, 4.0, 4.0)

# 1D marginal-of-W cache parameters.  Computed by 2D quadrature on a
# pair (z_i, z_j); since the same quartic coupling is used for each
# pair when lambdas are equal, the per-coord marginal is the same.  We
# still build one table per pair to support unequal lambdas if needed.
_MARG_GRIDSIZE: int = 500
_MARG_SPAN: float = 8.0


# ---------------------------------------------------------------------------
# Quadrature: build 1D marginal of an active coord under the quartic-coupled
# pair density.  Same shape for each pair (depends only on lambda_k).
# ---------------------------------------------------------------------------

def _build_pair_marginal_table(
    lam: float,
    gridsize: int = _MARG_GRIDSIZE,
    span: float = _MARG_SPAN,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 1D marginal of an active pair coord at coupling lambda.

    Pair density (unnormalized):
        q(z_i, z_j) = exp(-0.5 (z_i^2 + z_j^2) - lambda * ((z_i + z_j)/sqrt 2)^4 / 12)

    Returns
    -------
    Z_pair : float
        Normalizer 2D integral of q over the grid.
    grid : (gridsize,) ndarray
        Common 1D grid for both coord axes.
    pdf : (gridsize,) ndarray
        Marginal pdf p(z) = (integral of q over z_j) / Z_pair, evaluated
        at grid points.
    log_pdf : (gridsize,) ndarray
        log(pdf), clipped to avoid log(0) at the grid edges.
    cdf : (gridsize,) ndarray
        Empirical CDF on the grid, clipped to (1e-12, 1-1e-12) to support
        Phi-inverse calls without overflow.
    """
    grid = np.linspace(-span, span, gridsize)
    dz = grid[1] - grid[0]
    Z2, Z1 = np.meshgrid(grid, grid, indexing="ij")
    pair_pts = np.stack([Z2, Z1], axis=-1)
    zi, zj = pair_pts[..., 0], pair_pts[..., 1]
    u = (zi + zj) / np.sqrt(2.0)
    log_q = -0.5 * (zi * zi + zj * zj) - lam * u ** 4 / 12.0

    m = log_q.max()
    Z_pair = float(np.exp(m) * np.exp(log_q - m).sum() * dz * dz)
    pdf = (np.exp(log_q).sum(axis=1) * dz) / Z_pair
    log_pdf = np.log(np.maximum(pdf, 1e-30))
    cdf = np.cumsum(pdf) * dz
    cdf = np.clip(cdf, 1e-12, 1.0 - 1e-12)
    return Z_pair, grid, pdf, log_pdf, cdf


# Build per-pair tables at module load (cheap: <30ms for the 3 pairs).
_PAIR_TABLES = [_build_pair_marginal_table(lam) for lam in LAMBDAS_PPG]

# Joint normalizer of the full PPG density in W-space:
#   Z_joint = (2pi)^{(d-6)/2} * prod_k Z_pair_k
_LOG_Z_JOINT = (
    sum(np.log(table[0]) for table in _PAIR_TABLES)
    + 0.5 * (D_PPG - 2 * N_PAIRS) * np.log(2.0 * np.pi)
)

# Precomputed constants used in the hot path.
_PPG_INV_12 = 1.0 / 12.0
_PPG_INV_SQRT_2 = 1.0 / np.sqrt(2.0)
_PPG_LOG_2PI = float(np.log(2.0 * np.pi))


def _eval_pair_log_pdf(z: np.ndarray, k: int) -> np.ndarray:
    """log p(z) for active coord in pair k, vectorized."""
    _, grid, _, log_pdf, _ = _PAIR_TABLES[k]
    return np.interp(z, grid, log_pdf, left=-1e10, right=-1e10)


def _eval_pair_cdf(z: np.ndarray, k: int) -> np.ndarray:
    """F(z) for active coord in pair k, vectorized."""
    _, grid, _, _, cdf = _PAIR_TABLES[k]
    return np.interp(z, grid, cdf)


def _eval_pair_inv_cdf(u: np.ndarray, k: int) -> np.ndarray:
    """F^{-1}(u) for active coord in pair k, vectorized."""
    _, grid, _, _, cdf = _PAIR_TABLES[k]
    u_clip = np.clip(u, cdf[0], cdf[-1])
    # cdf is monotonically increasing, so np.interp inverts it cleanly.
    return np.interp(u_clip, cdf, grid)


# ---------------------------------------------------------------------------
# Sampling (rejection against base Gaussian envelope)
# ---------------------------------------------------------------------------

def _sample_W(N: int, rng: np.random.Generator, max_oversample: float = 20.0) -> np.ndarray:
    """Sample N points from f(W) by rejection against the N(0, I) envelope.

    Since f(W) = exp(-0.5 |W|^2) * exp(-pert(W)) / Z and pert(W) >= 0
    (each lambda_k > 0 and phi >= 0), the ratio f / N(0,I) <= (2pi)^{d/2} / Z
    with equality at W = 0.  Acceptance ratio = exp(-pert(W)) for each draw.
    """
    accepted: list[np.ndarray] = []
    n_have = 0
    n_total = 0
    n_max = max(int(N * max_oversample), 10000)
    while n_have < N:
        batch_size = max(N - n_have, 1024)
        candidates = rng.standard_normal((batch_size, D_PPG))
        # Compute pert(W) batch-vectorized
        pert = np.zeros(batch_size)
        for k in range(N_PAIRS):
            u = (candidates[:, 2 * k] + candidates[:, 2 * k + 1]) * _PPG_INV_SQRT_2
            pert += LAMBDAS_PPG[k] * u ** 4 * _PPG_INV_12
        u_rand = rng.uniform(size=batch_size)
        keep_mask = u_rand < np.exp(-pert)
        kept = candidates[keep_mask]
        accepted.append(kept)
        n_have += kept.shape[0]
        n_total += batch_size
        # Only treat acceptance as a failure if we've produced very few samples
        # *relative to what we asked for* and yet hit the oversample budget.
        if n_total > n_max and n_have < N:
            raise RuntimeError(
                f"PPG rejection sampler failed: got "
                f"{n_have}/{N} samples in {n_total} tries "
                f"(acceptance = {n_have / n_total:.3f})"
            )
    return np.concatenate(accepted)[:N]


def sample_ppg_noise(N: int, rng: np.random.Generator) -> np.ndarray:
    """Draw N samples from the PPG noise law on R^{D_PPG}.

    Returns eta of shape (N, d), where eta_j = logit(Phi(z_j)) and
    z_j is the rank-Gaussianized version of W_j.

    The rank-Gauss transform whitens the active-coord marginals from
    the quartic-perturbed shape to N(0, 1); padding coords are already
    N(0, 1) so the transform is the identity there.

    Numerically stable: uses logit_phi (no clipping at the logit-Phi step).
    The per-coord rank-Gauss step (active coords) still clips u to (1e-12,
    1 - 1e-12) since the pair-marginal CDF table has bounded support [-8, 8]
    and z = Phi^{-1}(u) for u at the table boundary is the correct
    truncation.
    """
    W = _sample_W(N, rng)
    # Per-coord rank-Gauss: z_j = Phi^{-1}(F_j(W_j))
    z = W.copy()
    for k in range(N_PAIRS):
        u_i = _eval_pair_cdf(W[:, 2 * k], k)
        u_j = _eval_pair_cdf(W[:, 2 * k + 1], k)
        z[:, 2 * k] = stats.norm.ppf(np.clip(u_i, 1e-12, 1.0 - 1e-12))
        z[:, 2 * k + 1] = stats.norm.ppf(np.clip(u_j, 1e-12, 1.0 - 1e-12))
    # Padding coords (k >= 2*N_PAIRS): already N(0,1), no transform.
    # eta = logit(Phi(z)) computed stably (no clipping).
    return logit_phi(z)


# ---------------------------------------------------------------------------
# Log-density evaluation
# ---------------------------------------------------------------------------

def log_ppg_density(eta: np.ndarray) -> np.ndarray:
    """Log-density of the PPG noise at points eta in R^{N x d}.

    Chain: eta → z (componentwise inv_logit_phi) → W (per-coord
    rank-Gauss inverse for active coords; identity for padding) → log f_W(W).

    Density:
        log pi_eta(eta) = log f_W(W) + sum_j log |dW_j/deta_j|

    Active coord j in 0..5:
        eta_j = logit(F_j(W_j))   (chain through z_j = Phi^{-1}(F_j(W_j)))
        deta_j/dW_j = (1/(u(1-u))) * f_j(W_j),   u = F_j(W_j) = sigm(eta_j)
        log|dW_j/deta_j| = log u + log(1-u) - log f_j(W_j)

    Padding coord j >= 6: z_j = W_j, eta_j = logit(Phi(W_j)) directly.
        log|dW_j/deta_j| = log_ndtr(W_j) + log_ndtr(-W_j) + 0.5 W_j^2 + 0.5 log(2 pi)
        (computed by logit_phi_log_jacobian_z).

    Stable: uses inv_logit_phi for eta -> z (no clipping). The active-coord
    u = sigm(eta) still uses the per-coord CDF table inversion, which
    requires u clipped to the table support; this is appropriate since the
    table itself is bounded.
    """
    eta = np.asarray(eta, dtype=np.float64)
    # Stable: eta -> z via inv_logit_phi (no clipping), then z -> u = Phi(z) -> W.
    z = inv_logit_phi(eta)
    # For active coords: u = Phi(z) = sigmoid(eta); need to invert pair-marginal
    # CDF. We compute u via sigmoid(eta) directly, with bounded clipping for the
    # table-bounded inverse.
    sig = 1.0 / (1.0 + np.exp(-eta))
    # The PPG pair-marginal CDF table is bounded, so for inversion we still
    # clip u to (1e-12, 1 - 1e-12). This is a real bound of the empirical
    # CDF table, not an arbitrary precision cap.
    np.clip(sig, 1e-12, 1.0 - 1e-12, out=sig)

    # Active coords: W_j = F_j^{-1}(sig_j) via pair tables
    W = np.empty_like(eta)
    for k in range(N_PAIRS):
        W[:, 2 * k] = _eval_pair_inv_cdf(sig[:, 2 * k], k)
        W[:, 2 * k + 1] = _eval_pair_inv_cdf(sig[:, 2 * k + 1], k)
    # Padding coords: W_j = z_j  (since z = Phi^{-1}(sig) = W for padding)
    W[:, 2 * N_PAIRS:] = z[:, 2 * N_PAIRS:]

    # log f_W(W) up to the joint normalizer -log Z_joint
    quad = -0.5 * (W * W).sum(axis=-1)
    pert = np.zeros(eta.shape[0])
    for k in range(N_PAIRS):
        u_pair = (W[:, 2 * k] + W[:, 2 * k + 1]) * _PPG_INV_SQRT_2
        pert += LAMBDAS_PPG[k] * u_pair ** 4 * _PPG_INV_12
    log_f_W = quad - pert - _LOG_Z_JOINT

    # log|dW/deta| sum over coords
    # Active coords: log sig + log(1-sig) - log f_pair(W)
    log_jac = np.log(sig).sum(axis=-1) + np.log1p(-sig).sum(axis=-1)
    # Subtract sum of log marginals at active coords
    for k in range(N_PAIRS):
        log_jac -= _eval_pair_log_pdf(W[:, 2 * k], k)
        log_jac -= _eval_pair_log_pdf(W[:, 2 * k + 1], k)
    # Padding coords (z_j = W_j): contribute log|dW_j/deta_j| computed stably
    # via the analytic identity for the logit-Phi transform.
    # We need to subtract the log sig + log(1-sig) we double-counted above
    # for padding coords (since the unified `log_jac` summed it for ALL coords,
    # but padding has a different Jacobian form), and add the stable form.
    # Cleaner: redo the log_jac calculation per coord type.
    # Reset and recompute:
    log_jac_active = np.zeros(eta.shape[0])
    for k in range(N_PAIRS):
        # Active coord j = 2k or 2k+1: log|dW/deta| = log u + log(1-u) - log f_j(W)
        ui = sig[:, 2 * k]
        uj = sig[:, 2 * k + 1]
        log_jac_active += (
            np.log(ui) + np.log1p(-ui)
            + np.log(uj) + np.log1p(-uj)
            - _eval_pair_log_pdf(W[:, 2 * k], k)
            - _eval_pair_log_pdf(W[:, 2 * k + 1], k)
        )
    # Padding coord j >= 2 N_PAIRS: stable analytical Jacobian on z_j = W_j.
    log_jac_padding = logit_phi_log_jacobian_z(W[:, 2 * N_PAIRS:]).sum(axis=-1)
    log_jac = log_jac_active + log_jac_padding

    return log_f_W + log_jac


# ---------------------------------------------------------------------------
# Oracle subspace: V_KL = top-r eigenvectors of population score covariance C
# ---------------------------------------------------------------------------
#
# By the axis-pair-aligned construction, the rank-Gaussianized copula
# density depends nontrivially only on coords 0..5 (the active pairs).
# Hence C has EXACT rank 6 with support span{e_0, ..., e_5}.
#
# Within that 6-dim subspace, C has a 3+3 eigenstructure:
#   Top 3 eigvecs = span{b_0, b_1, b_2}   (primary, eigval ~0.5)
#   Next 3 eigvecs = span{b_0_perp, b_1_perp, b_2_perp}  (secondary, eigval ~0.12)
# with b_k = (e_{2k} + e_{2k+1})/sqrt(2) and b_k_perp = (e_{2k} - e_{2k+1})/sqrt(2).
#
# For r = 3: V_KL = span{b_0, b_1, b_2}  -- moderate cliff.
# For r = 6: V_KL = span{e_0, ..., e_5}  -- hard cliff (full active subspace).
# For 3 < r < 6: V_KL extends by including secondary directions in order
#                (b_0_perp first, then b_1_perp, then b_2_perp).
#
# Verified by FD cross-check at N=30,000 samples: sin Theta < 0.01 at r=3,
# exactly 0 at r=6.

def oracle_subspace_ppg(r: int) -> np.ndarray:
    """V_KL: top-r eigenspace of the population score covariance C.

    See module docstring for the analytical form.  For r in {3, 4, 5, 6}
    returns the analytical V_KL; r > 6 raises since C has exact rank 6.
    """
    if r > 2 * N_PAIRS:
        raise ValueError(
            f"oracle_subspace_ppg: r={r} exceeds active rank "
            f"r* = {2 * N_PAIRS}"
        )
    if r < 1:
        raise ValueError(f"oracle_subspace_ppg: r={r} must be >= 1")

    # Primary directions: b_k = (e_{2k} + e_{2k+1}) / sqrt 2
    b_primary = np.zeros((D_PPG, N_PAIRS))
    for k in range(N_PAIRS):
        b_primary[2 * k, k] = _PPG_INV_SQRT_2
        b_primary[2 * k + 1, k] = _PPG_INV_SQRT_2

    # Secondary directions: b_k_perp = (e_{2k} - e_{2k+1}) / sqrt 2
    b_secondary = np.zeros((D_PPG, N_PAIRS))
    for k in range(N_PAIRS):
        b_secondary[2 * k, k] = _PPG_INV_SQRT_2
        b_secondary[2 * k + 1, k] = -_PPG_INV_SQRT_2

    # Assemble in order: all primary, then secondary as needed.
    if r <= N_PAIRS:
        V = b_primary[:, :r]
    else:
        V = np.concatenate([b_primary, b_secondary[:, :r - N_PAIRS]], axis=1)

    # Orthonormalize (should already be ON by construction, but defensive QR).
    V, _ = np.linalg.qr(V)
    return V
