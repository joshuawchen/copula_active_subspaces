"""Cusp-banana noise law on d = 20 coordinates: two axis-aligned pairs with
W_{2k+1} = alpha (|W_{2k}| - sqrt(2 / pi)) + sigma eps_k, alpha = 0.8, whose
copula score has an infinite Hermite expansion.
"""
from __future__ import annotations

import numpy as np

from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

D_CUSP: int = 20
R_CUSP_TARGET: int = 4
N_PAIRS: int = 2  # cusp pairs at coords (0,1), (2,3)

# Cusp dependence strength.  At alpha = 0.8 the centered-|W_1| term carries
# alpha^2 (1 - 2/pi) ~ 0.23 of the variance of W_{2k+1}; strong enough for a
# clean rank-4 cliff while keeping sigma_eps^2 = 1 - 0.23 > 0.
ALPHA_CUSP: float = 0.8

_C_ABS: float = np.sqrt(2.0 / np.pi)                 # E|N(0,1)|
_VAR_ABS: float = 1.0 - 2.0 / np.pi                  # Var(|N(0,1)|)
_SIGMA_EPS2: float = 1.0 - ALPHA_CUSP ** 2 * _VAR_ABS
_SIGMA_EPS: float = float(np.sqrt(_SIGMA_EPS2))
assert _SIGMA_EPS2 > 0.0, "alpha too large: sigma_eps^2 <= 0"

_LOG_2PI: float = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Public sampler and density
# ---------------------------------------------------------------------------

def sample_cusp_banana_noise(N: int, rng: np.random.Generator) -> np.ndarray:
    """Draw N samples from the cusp-banana noise law on R^{D_CUSP}.

    Returns eta of shape (N, d).  Stable transform: logit_phi (no clipping).
    """
    W = rng.standard_normal((N, D_CUSP))
    for k in range(N_PAIRS):
        w1 = W[:, 2 * k]
        eps = W[:, 2 * k + 1]
        W[:, 2 * k + 1] = (
            ALPHA_CUSP * (np.abs(w1) - _C_ABS) + _SIGMA_EPS * eps
        )
    # z = W (L = I), eta = logit(Phi(z)) computed stably (no clipping).
    return logit_phi(W)


def log_cusp_banana_density(eta: np.ndarray) -> np.ndarray:
    """Log-density of the cusp-banana noise at points eta in R^{N x d}.

    Chain: eta -> z (= W, since L = I) -> log pi_W(W) + log|dW/deta|.

      eta_j = logit(Phi(W_j))   componentwise,
      W_j   = Phi^{-1}(sigm(eta_j)).

    pi_W = prod_k [ phi(W_{2k}) * phi((W_{2k+1}-mu_k)/sigma_eps)/sigma_eps ]
           * prod_{j>=4} phi(W_j),
    with mu_k = alpha (|W_{2k}| - c_abs).
    """
    eta = np.asarray(eta, dtype=np.float64)
    W = inv_logit_phi(eta)  # stable inverse, no clipping

    w_first = W[:, 0:2 * N_PAIRS:2]     # (N, N_PAIRS) — coords 0, 2
    w_second = W[:, 1:2 * N_PAIRS:2]    # (N, N_PAIRS) — coords 1, 3
    mu = ALPHA_CUSP * (np.abs(w_first) - _C_ABS)
    resid = w_second - mu
    log_pi_pairs = (
        -0.5 * (w_first * w_first).sum(axis=1)
        - 0.5 * (resid * resid).sum(axis=1) / _SIGMA_EPS2
        - 0.5 * N_PAIRS * np.log(_SIGMA_EPS2)
        - N_PAIRS * _LOG_2PI
    )

    W_rest = W[:, 2 * N_PAIRS:]
    n_rest = D_CUSP - 2 * N_PAIRS
    log_pi_rest = (
        -0.5 * (W_rest * W_rest).sum(axis=1)
        - 0.5 * n_rest * _LOG_2PI
    )
    log_pi_W = log_pi_pairs + log_pi_rest

    log_jac = logit_phi_log_jacobian_z(W).sum(axis=1)
    return log_pi_W + log_jac


# ---------------------------------------------------------------------------
# Oracle subspace: exact rank-r* = 2*N_PAIRS, support span{e_0,...,e_{r*-1}}
# ---------------------------------------------------------------------------

def oracle_subspace_cusp_banana(r: int) -> np.ndarray:
    """V_KL: top-r eigenspace of the population score covariance C.

    By the axis-aligned construction, C has exact rank 2*N_PAIRS with support
    span{e_0, ..., e_{2*N_PAIRS-1}}.  For r <= 2*N_PAIRS we return the canonical
    basis [e_0, ..., e_{r-1}].
    """
    if r > 2 * N_PAIRS:
        raise ValueError(
            f"oracle_subspace_cusp_banana: r={r} exceeds active rank "
            f"r* = {2 * N_PAIRS}"
        )
    V = np.zeros((D_CUSP, r))
    for j in range(r):
        V[j, j] = 1.0
    return V
