"""log pi_U(u) for U = V_r^T Z by Laplace-centred multi-start importance sampling
over the complement (log_pi_U_via_marginalization), and draws of U
(sample_pi_U). Examples are dispatched by name through _get_active_callables.
"""
from __future__ import annotations
import time
from typing import Callable, Optional

import numpy as np
from scipy import stats


# =====================================================================
# Per-benchmark closed-form active-coordinate density
# =====================================================================
#
# Each benchmark exposes a callables-dict via _get_active_callables(name).
# The contract is:
#
#   d                  : int, ambient dimension
#   k_act              : int, number of "active" W coordinates (non-Gaussian)
#   L                  : (d, d) Cholesky of latent covariance Σ_0
#   L_inv              : (d, d) inverse Cholesky (precomputed)
#   log_det_L          : float, log|det L|
#   log_pi_W_active    : callable (N, k_act) -> (N,)
#   grad_log_pi_W_active : callable (N, k_act) -> (N, k_act)
#   hess_log_pi_W_active : callable (N, k_act) -> (N, k_act, k_act)
#   sample_noise       : callable (N, rng) -> (N, d), draws η from π_n
#
# The active coordinates are the first k_act of W (the latent driver).
# The remaining d - k_act are i.i.d. standard normal.
#
# Banana (Example 1): conformal z^2 block at coords 0..3 (2 inputs W0,W1
#   and 2 outputs W2,W3), k_act = 4.
# Cubic_banana: 3 banana pairs at coords 0..5, k_act = 6.
# PPG: 3 quartic-perturbed axis-pairs at coords 0..5, k_act = 6.
# =====================================================================


def _banana_callables() -> dict:
    """Closed-form callables for the Example-1 conformal z^2 noise (§5).

    Active-block density, score, and Hessian are imported from cas.noise
    (the single source of truth for the law) rather than re-derived here, so
    they track the deployed law automatically and cannot drift from the
    sampler.
    """
    from .config import D_OBS, K_BAN
    from .noise import (
        L_CHOL, sample_noise,
        log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
    )

    k_act = 2 * K_BAN
    L = L_CHOL
    L_inv = np.linalg.inv(L)
    log_det_L = float(np.log(np.abs(np.diag(L))).sum())

    return {
        "d": D_OBS,
        "k_act": k_act,
        "L": L,
        "L_inv": L_inv,
        "log_det_L": log_det_L,
        "log_pi_W_active": log_pi_W_active,
        "grad_log_pi_W_active": grad_log_pi_W_active,
        "hess_log_pi_W_active": hess_log_pi_W_active,
        "sample_noise": sample_noise,
    }





def _cubic_banana_callables() -> dict:
    """Closed-form callables for the cubic-banana noise of §SM3.

    Three banana pairs at coords 0..5; padding at coords 6..D_CB-1.
    Per-pair conditional:
        W_{2k}   ~ N(0, 1)
        W_{2k+1} | W_{2k} ~ N(beta * W_{2k}^3, s^2)
    where beta = ALPHA_CB / sqrt(15), s^2 = 1 - ALPHA_CB^2. No Toeplitz
    mixing: L = I, log|det L| = 0.
    """
    from .cubic_banana import (
        D_CB, ALPHA_CB, N_PAIRS, sample_cubic_banana_noise,
    )

    beta = ALPHA_CB / np.sqrt(15.0)
    s2 = 1.0 - ALPHA_CB ** 2
    k_act = 2 * N_PAIRS  # = 6
    L = np.eye(D_CB)
    L_inv = np.eye(D_CB)
    log_det_L = 0.0

    def log_pi_W_active(W_act):
        """log π for N_PAIRS cubic-banana pairs (no Gaussian residual)."""
        W_act = np.atleast_2d(W_act)
        log_p = np.zeros(W_act.shape[0])
        for j in range(N_PAIRS):
            W1 = W_act[:, 2 * j]
            W2 = W_act[:, 2 * j + 1]
            log_p = log_p - 0.5 * W1 ** 2 - 0.5 * np.log(2 * np.pi)
            mean2 = beta * W1 ** 3
            log_p = (log_p
                     - 0.5 * (W2 - mean2) ** 2 / s2
                     - 0.5 * np.log(2 * np.pi * s2))
        return log_p

    def grad_log_pi_W_active(W_act):
        """∇log π wrt W_act, returns (N, k_act)."""
        W_act = np.atleast_2d(W_act)
        grad = np.zeros_like(W_act)
        for j in range(N_PAIRS):
            W1 = W_act[:, 2 * j]
            W2 = W_act[:, 2 * j + 1]
            resid = W2 - beta * W1 ** 3
            # ∂/∂W1 = -W1 + 3 beta W1^2 resid / s^2
            grad[:, 2 * j] = -W1 + 3.0 * beta * W1 ** 2 * resid / s2
            # ∂/∂W2 = -resid / s^2
            grad[:, 2 * j + 1] = -resid / s2
        return grad

    def hess_log_pi_W_active(W_act):
        """Hessian wrt W_act, returns (N, k_act, k_act)."""
        W_act = np.atleast_2d(W_act)
        N = W_act.shape[0]
        H = np.zeros((N, k_act, k_act))
        for j in range(N_PAIRS):
            W1 = W_act[:, 2 * j]
            W2 = W_act[:, 2 * j + 1]
            resid = W2 - beta * W1 ** 3
            # ∂²/∂W1² = -1 + 6 beta W1 resid / s^2 - 9 beta^2 W1^4 / s^2
            H[:, 2 * j, 2 * j] = (
                -1.0
                + 6.0 * beta * W1 * resid / s2
                - 9.0 * beta ** 2 * W1 ** 4 / s2
            )
            # ∂²/∂W2² = -1/s^2
            H[:, 2 * j + 1, 2 * j + 1] = -1.0 / s2
            # ∂²/∂W1∂W2 = 3 beta W1^2 / s^2
            H[:, 2 * j, 2 * j + 1] = 3.0 * beta * W1 ** 2 / s2
            H[:, 2 * j + 1, 2 * j] = 3.0 * beta * W1 ** 2 / s2
        return H

    return {
        "d": D_CB,
        "k_act": k_act,
        "L": L,
        "L_inv": L_inv,
        "log_det_L": log_det_L,
        "log_pi_W_active": log_pi_W_active,
        "grad_log_pi_W_active": grad_log_pi_W_active,
        "hess_log_pi_W_active": hess_log_pi_W_active,
        "sample_noise": sample_cubic_banana_noise,
    }


def _ppg_callables() -> dict:
    """Closed-form callables for the PPG noise of §SM3.

    Three quartic-perturbed axis-pairs at coords 0..5; padding at coords
    6..D_PPG-1. The active marginal density is:

        π_W^act(W_{0..5}) = (1/Z_pairs) * exp(-0.5 ||W||² - sum_k λ_k u_k^4/12)

    where u_k = (W_{2k} + W_{2k+1})/sqrt(2), λ_k = LAMBDAS_PPG[k], and
    Z_pairs = prod_k Z_{pair,k} is precomputed in cas.ppg._PAIR_TABLES.
    No Toeplitz mixing: L = I, log|det L| = 0.
    """
    from .ppg import (
        D_PPG, LAMBDAS_PPG, N_PAIRS, _PAIR_TABLES,
        _PPG_INV_SQRT_2, sample_ppg_noise,
    )

    k_act = 2 * N_PAIRS  # = 6
    L = np.eye(D_PPG)
    L_inv = np.eye(D_PPG)
    log_det_L = 0.0
    lambdas = np.array(LAMBDAS_PPG)

    # Pre-sum log Z_pairs (the normalizer of the active joint).
    log_Z_pairs = float(sum(np.log(table[0]) for table in _PAIR_TABLES))

    def log_pi_W_active(W_act):
        """log π for the 6-dim active block under PPG.

        π_W^act(W) = (1/Z_pairs) exp(-0.5 ||W||² - sum_k λ_k u_k^4/12)
        """
        W_act = np.atleast_2d(W_act)
        N = W_act.shape[0]
        quad = -0.5 * (W_act * W_act).sum(axis=-1)
        pert = np.zeros(N)
        for k in range(N_PAIRS):
            u_k = (W_act[:, 2 * k] + W_act[:, 2 * k + 1]) * _PPG_INV_SQRT_2
            pert = pert + lambdas[k] * u_k ** 4 / 12.0
        return quad - pert - log_Z_pairs

    def grad_log_pi_W_active(W_act):
        """∇log π wrt W_act.

        ∂/∂W_{2k}   = -W_{2k}   - λ_k u_k^3 / (3 sqrt 2)
        ∂/∂W_{2k+1} = -W_{2k+1} - λ_k u_k^3 / (3 sqrt 2)
        """
        W_act = np.atleast_2d(W_act)
        grad = -W_act.copy()
        for k in range(N_PAIRS):
            u_k = (W_act[:, 2 * k] + W_act[:, 2 * k + 1]) * _PPG_INV_SQRT_2
            pert_term = lambdas[k] * u_k ** 3 / (3.0 * np.sqrt(2.0))
            grad[:, 2 * k] = grad[:, 2 * k] - pert_term
            grad[:, 2 * k + 1] = grad[:, 2 * k + 1] - pert_term
        return grad

    def hess_log_pi_W_active(W_act):
        """Hessian wrt W_act.

        ∂²/∂W_{2k}² = ∂²/∂W_{2k+1}² = -1 - λ_k u_k² / 2
        ∂²/∂W_{2k}∂W_{2k+1} = -λ_k u_k² / 2
        Off-pair entries are 0.
        """
        W_act = np.atleast_2d(W_act)
        N = W_act.shape[0]
        H = np.zeros((N, k_act, k_act))
        # Identity Hessian piece: -I from -0.5 ||W||^2
        for i in range(k_act):
            H[:, i, i] = -1.0
        for k in range(N_PAIRS):
            u_k = (W_act[:, 2 * k] + W_act[:, 2 * k + 1]) * _PPG_INV_SQRT_2
            curv = -lambdas[k] * u_k ** 2 / 2.0
            H[:, 2 * k, 2 * k] += curv
            H[:, 2 * k + 1, 2 * k + 1] += curv
            H[:, 2 * k, 2 * k + 1] += curv
            H[:, 2 * k + 1, 2 * k] += curv
        return H

    return {
        "d": D_PPG,
        "k_act": k_act,
        "L": L,
        "L_inv": L_inv,
        "log_det_L": log_det_L,
        "log_pi_W_active": log_pi_W_active,
        "grad_log_pi_W_active": grad_log_pi_W_active,
        "hess_log_pi_W_active": hess_log_pi_W_active,
        "sample_noise": sample_ppg_noise,
    }


def _even_fold_callables() -> dict:
    """Closed-form callables for the even-fold supplement noise.

    Active-block density, score, and Hessian are imported from cas.even_fold
    (the single source of truth) so they track the deployed law and cannot
    drift from the sampler.
    """
    from .even_fold import (
        D_EF, N_PAIRS_EF, L_CHOL_EF, sample_even_fold_noise,
        log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
    )

    k_act = 2 * N_PAIRS_EF
    L = L_CHOL_EF
    L_inv = np.linalg.inv(L)
    log_det_L = float(np.log(np.abs(np.diag(L))).sum())

    return {
        "d": D_EF,
        "k_act": k_act,
        "L": L,
        "L_inv": L_inv,
        "log_det_L": log_det_L,
        "log_pi_W_active": log_pi_W_active,
        "grad_log_pi_W_active": grad_log_pi_W_active,
        "hess_log_pi_W_active": hess_log_pi_W_active,
        "sample_noise": sample_even_fold_noise,
    }


def _conformal_cube_callables() -> dict:
    """Closed-form callables for the conformal z^3 supplement noise.

    Active-block density, score, and Hessian are imported from
    cas.conformal_cube (the single source of truth) so they track the
    deployed law and cannot drift from the sampler.
    """
    from .conformal_cube import (
        D_C3, L_CHOL_C3, sample_conformal_cube_noise,
        log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
    )

    k_act = 4
    L = L_CHOL_C3
    L_inv = np.linalg.inv(L)
    log_det_L = float(np.log(np.abs(np.diag(L))).sum())

    return {
        "d": D_C3,
        "k_act": k_act,
        "L": L,
        "L_inv": L_inv,
        "log_det_L": log_det_L,
        "log_pi_W_active": log_pi_W_active,
        "grad_log_pi_W_active": grad_log_pi_W_active,
        "hess_log_pi_W_active": hess_log_pi_W_active,
        "sample_noise": sample_conformal_cube_noise,
    }


# Module-level dispatch. Each value is a thunk so we don't import cas.noise
# (and its dependencies) until needed.
_BENCHMARK_DISPATCH: dict = {
    "banana": _banana_callables,
    "cubic_banana": _cubic_banana_callables,
    "ppg":          _ppg_callables,
    "even_fold":    _even_fold_callables,
    "conformal_cube": _conformal_cube_callables,
}


def _get_active_callables(benchmark: str) -> dict:
    """Lookup + materialize the active-coordinate callables for a benchmark.

    Cached per-benchmark; first call instantiates, subsequent calls return
    the same dict.
    """
    if benchmark not in _BENCHMARK_DISPATCH:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Known: "
            f"{list(_BENCHMARK_DISPATCH.keys())}. See docs/BENCHMARK_PORTING.md "
            f"for how to add a new one."
        )
    cache_key = f"_callables_cache_{benchmark}"
    if not hasattr(_get_active_callables, cache_key):
        setattr(
            _get_active_callables, cache_key,
            _BENCHMARK_DISPATCH[benchmark](),
        )
    return getattr(_get_active_callables, cache_key)


# =====================================================================
# Newton mode finder
# =====================================================================

def _find_mode(
    u_tilde_act: np.ndarray,    # (k_act,)
    u_tilde_res: np.ndarray,    # (d - k_act,)
    w_tilde_act: np.ndarray,    # (k_act, dm) where dm = d - r
    w_tilde_res: np.ndarray,    # (d - k_act, dm)
    w0: np.ndarray,             # (dm,) initial guess
    grad_log_pi_W_active,
    hess_log_pi_W_active,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Newton iteration to find w* = argmax log f(w), where

        log f(w) = log π_act(u_act + w̃_act w) + log γ_res(u_res + w̃_res w)

    ∇ log f(w) = w̃_act^T ∇_a log π_act - w̃_res^T (u_res + w̃_res w)
    H log f(w) = w̃_act^T H_act w̃_act - w̃_res^T w̃_res

    Negative-definiteness is enforced by shifting the Hessian if any
    eigenvalue is non-negative — this can happen at sin Θ ≈ 0 where the
    Hessian becomes (numerically) singular.
    """
    w = w0.copy()
    G_res = w_tilde_res.T @ w_tilde_res  # constant
    dm = w.shape[0]

    for _ in range(max_iter):
        a = u_tilde_act + w_tilde_act @ w
        grad_act = grad_log_pi_W_active(a[None, :])[0]
        H_act = hess_log_pi_W_active(a[None, :])[0]

        grad = (w_tilde_act.T @ grad_act
                - w_tilde_res.T @ (u_tilde_res + w_tilde_res @ w))
        H = w_tilde_act.T @ H_act @ w_tilde_act - G_res

        # Ensure H is negative-definite (we're at a max).
        eigs = np.linalg.eigvalsh(H)
        if eigs.max() > -1e-8:
            shift = -(eigs.max() + 1e-3)
            H = H - shift * np.eye(dm)

        step = -np.linalg.solve(H, grad)
        w_new = w + step
        if np.linalg.norm(step) < tol:
            return w_new
        w = w_new
    return w


def _log_integrand_at(w, u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
                       log_pi_W_active):
    """Evaluate log f(w) for comparison across starting points."""
    a = u_tilde_act + w_tilde_act @ w
    log_act = float(log_pi_W_active(a[None, :])[0])
    b = u_tilde_res + w_tilde_res @ w
    log_gauss = -0.5 * float((b ** 2).sum()) - 0.5 * len(b) * np.log(2 * np.pi)
    return log_act + log_gauss


def _multistart_mode(
    u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
    mu_gauss, Sigma_gauss,
    log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
    n_starts: int, seed: int,
):
    """Multi-start Newton: try n_starts starting points, return the best mode.

    Starting points:
      [0]   Gaussian-only mode (the original Laplace start, often correct)
      [1:]  Random perturbations of the Gaussian mode with std 2x its
            covariance — large enough to escape spurious local maxima but
            not so large that we waste iterations on hopeless starts.

    Returns (best_w, best_log_f, n_starts_used) where n_starts_used is the
    1-indexed iteration at which the best mode was first found. All n_starts
    are still evaluated for safety; n_starts_used is purely diagnostic.
    """
    rng = np.random.default_rng(seed)
    dm = mu_gauss.shape[0]

    # Robust PSD-ize Sigma_gauss before Cholesky: np.linalg.inv can return
    # a slightly non-symmetric matrix with tiny-negative eigenvalues from
    # rounding; +1e-8*I alone is sometimes insufficient. Symmetrize, then
    # shift to ensure min-eigenvalue >= 1e-6.
    Sg = 0.5 * (Sigma_gauss + Sigma_gauss.T)
    eigs_sg = np.linalg.eigvalsh(Sg)
    if eigs_sg.min() < 1e-6:
        Sg = Sg + (1e-6 - eigs_sg.min()) * np.eye(dm)
    L_g = np.linalg.cholesky(Sg)

    best_w = None
    best_log_f = -np.inf
    best_at = 0
    for k in range(n_starts):
        if k == 0:
            start = mu_gauss
        else:
            start = mu_gauss + L_g @ rng.standard_normal(dm) * 2.0
        try:
            w_star = _find_mode(
                u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res, start,
                grad_log_pi_W_active, hess_log_pi_W_active,
            )
            log_f = _log_integrand_at(
                w_star, u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
                log_pi_W_active,
            )
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        if log_f > best_log_f:
            best_log_f = log_f
            best_w = w_star
            best_at = k + 1

    if best_w is None:
        raise RuntimeError(
            "All Newton starts failed; possibly degenerate Hessian. "
            "Try increasing n_starts or check that V_r and V_perp are "
            "well-conditioned."
        )
    return best_w, best_log_f, best_at


# =====================================================================
# Laplace-corrected IS estimator
# =====================================================================

def _laplace_is_one_u(
    u: np.ndarray,
    V_r: np.ndarray,
    V_perp: np.ndarray,
    callables: dict,
    M_aux: int,
    n_mode_starts: int,
    seed: int,
    want_grad: bool = False,
) -> dict:
    """One-u IS estimate of log π_U(u).

    Args:
        u:               (r,) point in projected coords
        V_r:             (d, r) subspace
        V_perp:          (d, d-r) orthogonal complement
        callables:       dict from _get_active_callables
        M_aux:           IS sample count
        n_mode_starts:   multi-start Newton restart count
        seed:            RNG seed for IS samples + Newton perturbations

    Returns:
        dict with:
            log_pi_U:        scalar estimate of log π_U(u)
            rel_se:           scalar relative SE of the IS estimator
            M_eff_frac:       effective sample size fraction
            log_f_at_mode:    log f(w*) at the chosen Laplace center
            n_starts_used:    how many Newton starts before the best was found
    """
    d = callables["d"]
    k_act = callables["k_act"]
    L_inv = callables["L_inv"]
    log_det_L = callables["log_det_L"]
    log_pi_W_active = callables["log_pi_W_active"]
    grad_log_pi_W_active = callables["grad_log_pi_W_active"]
    hess_log_pi_W_active = callables["hess_log_pi_W_active"]

    dm = V_perp.shape[1]  # d - r

    # Per-u-precomputable quantities
    u_tilde = L_inv @ (V_r @ u)
    w_tilde = L_inv @ V_perp
    w_tilde_act = w_tilde[:k_act, :]
    w_tilde_res = w_tilde[k_act:, :]
    u_tilde_act = u_tilde[:k_act]
    u_tilde_res = u_tilde[k_act:]

    # Initial Gaussian-only IS proposal (matches the residual γ factor).
    # When r < k_act (e.g. PPG at r=3, k_act=6), w_tilde_res is rank-defective
    # (shape (d-k_act, dm) with dm > d-k_act), so G_res = w_tilde_res^T
    # w_tilde_res has rank d-k_act < dm and inv(G_res) is numerically
    # degenerate. In that case the residual γ factor alone doesn't constrain
    # w; fall back to a standard-Gaussian start and let Newton find the
    # mode using the full Hessian.
    rank_full = (w_tilde_res.shape[0] >= dm)
    if rank_full:
        G_res = w_tilde_res.T @ w_tilde_res + 1e-10 * np.eye(dm)
        Sigma_gauss = np.linalg.inv(G_res)
        mu_gauss = -Sigma_gauss @ w_tilde_res.T @ u_tilde_res
    else:
        # Rank-deficient residual: use I as the start covariance, origin
        # as the start mean. Multi-start Newton handles the actual mode.
        Sigma_gauss = np.eye(dm)
        mu_gauss = np.zeros(dm)

    # Multi-start Newton to locate the integrand's mode
    w_star, log_f_best, n_starts_used = _multistart_mode(
        u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
        mu_gauss, Sigma_gauss,
        log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
        n_starts=n_mode_starts, seed=seed,
    )

    # Hessian at the mode → Laplace covariance
    a_star = u_tilde_act + w_tilde_act @ w_star
    H_act = hess_log_pi_W_active(a_star[None, :])[0]
    H_full = w_tilde_act.T @ H_act @ w_tilde_act - w_tilde_res.T @ w_tilde_res

    eigs_h = np.linalg.eigvalsh(H_full)
    if eigs_h.max() > -1e-6:
        H_full = H_full - (eigs_h.max() + 1e-3) * np.eye(dm)
    Sigma_q = np.linalg.inv(-H_full)
    Sigma_q = 0.5 * (Sigma_q + Sigma_q.T)
    eigs_q = np.linalg.eigvalsh(Sigma_q)
    if eigs_q.min() <= 0:
        Sigma_q = Sigma_q + (1e-8 - eigs_q.min()) * np.eye(dm)
    L_q = np.linalg.cholesky(Sigma_q)
    Sigma_q_inv = np.linalg.inv(Sigma_q)
    log_det_Sigma_q = np.linalg.slogdet(Sigma_q)[1]

    # Sample M_aux from q ~ N(w_star, Sigma_q)
    rng_is = np.random.default_rng(seed + 10000)
    w_samples = w_star + (L_q @ rng_is.standard_normal((dm, M_aux))).T

    # Integrand
    W_act_eval = u_tilde_act[None, :] + w_samples @ w_tilde_act.T
    log_active = log_pi_W_active(W_act_eval)
    W_res_eval = u_tilde_res[None, :] + w_samples @ w_tilde_res.T
    log_gauss_res = (
        -0.5 * (W_res_eval ** 2).sum(axis=1)
        - 0.5 * (d - k_act) * np.log(2 * np.pi)
    )
    log_integrand = log_active + log_gauss_res

    # Proposal density
    diff = w_samples - w_star
    log_q = (
        -0.5 * np.einsum("mi,ij,mj->m", diff, Sigma_q_inv, diff)
        - 0.5 * dm * np.log(2 * np.pi)
        - 0.5 * log_det_Sigma_q
    )

    log_w = log_integrand - log_q
    log_w_max = log_w.max()
    w_norm = np.exp(log_w - log_w_max)

    log_pi_U = log_w_max + np.log(w_norm.mean()) - log_det_L

    M_eff = (w_norm.sum() ** 2) / (w_norm ** 2).sum()
    rel_se = float(w_norm.std(ddof=1) / w_norm.mean() / np.sqrt(M_aux))

    out = {
        "log_pi_U": float(log_pi_U),
        "rel_se": rel_se,
        "M_eff_frac": float(M_eff / M_aux),
        "log_f_at_mode": float(log_f_best),
        "n_starts_used": int(n_starts_used),
    }

    if want_grad:
        # grad_u log pi_U(u) = E_{f(.|u)}[ grad_u log f(u, w) ], the
        # self-normalized IS average against the SAME weights computed above
        # (differentiate pi_U(u) = int f(u,w) dw under the integral sign).
        # With u_tilde = L^{-1} V_r u, the u-dependence of
        #   log f = log pi_act(u_tilde_act + w_tilde_act w)
        #           + log gamma(u_tilde_res + w_tilde_res w)
        # enters only through u_tilde, so by the chain rule
        #   grad_u log f = P_act^T grad_a log pi_act(a) - P_res^T b,
        # with P = L^{-1} V_r split at k_act, a = W_act_eval, b = W_res_eval.
        P = L_inv @ V_r                      # (d, r)
        P_act, P_res = P[:k_act, :], P[k_act:, :]
        G_act = grad_log_pi_W_active(W_act_eval)          # (M, k_act)
        grad_f = G_act @ P_act - W_res_eval @ P_res       # (M, r)
        wts = w_norm / w_norm.sum()
        out["grad_log_pi_U"] = (wts[None, :] @ grad_f).ravel()

    return out


# =====================================================================
# Public API
# =====================================================================

def _orthonormal_complement(V_r: np.ndarray) -> np.ndarray:
    """Given V_r ∈ R^{d × r} orthonormal, return V_⊥ ∈ R^{d × (d-r)} orthonormal."""
    d, r = V_r.shape
    M = np.eye(d) - V_r @ V_r.T
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :d - r]


def log_pi_U_via_marginalization(
    U: np.ndarray,
    V_r: np.ndarray,
    benchmark: str = "banana",
    *,
    M_aux: int = 100_000,
    n_mode_starts: int = 20,
    seed: int = 0,
    progress_every: Optional[int] = None,
) -> dict:
    """Compute log π_U(u) for each u in U via Laplace-corrected multi-start IS.

    Args:
        U: (N_outer, r) — query points in projected coords.
        V_r: (d, r) — orthonormal subspace.
        benchmark: noise-law key (see _BENCHMARK_DISPATCH).
        M_aux: IS sample count per u. Pilot recommends 10^5 for ≤1% rel SE
               at sin Θ ≤ 0.3; 10^4 is fine for purely diagnostic use.
        n_mode_starts: Newton multi-start count. 20 is the pilot-validated
                       default that fixes all the ~1% catastrophic cells at
                       sin Θ ≈ 0.65.
        seed: RNG seed; reproducible.
        progress_every: print a progress line every N u's (None = silent).

    Returns:
        dict with per-u arrays:
            log_pi_U:       (N_outer,) log π_U estimates
            rel_se:         (N_outer,) relative SE
            M_eff_frac:     (N_outer,) effective sample size fraction
            log_f_at_mode:  (N_outer,) log f(w*) at chosen Laplace center
            n_starts_used:  (N_outer,) Newton starts before best mode found
            wall_time_s:    total wall time
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    N_outer, r = U.shape

    V_r = np.asarray(V_r, dtype=float)
    if V_r.shape[1] != r:
        raise ValueError(
            f"V_r has r={V_r.shape[1]} but U has r={r}; "
            "they must match."
        )

    callables = _get_active_callables(benchmark)
    d = callables["d"]
    if V_r.shape[0] != d:
        raise ValueError(
            f"V_r has d={V_r.shape[0]} but benchmark '{benchmark}' has "
            f"d={d}; they must match."
        )

    V_perp = _orthonormal_complement(V_r)

    out_log_pi_U = np.empty(N_outer)
    out_rel_se = np.empty(N_outer)
    out_M_eff_frac = np.empty(N_outer)
    out_log_f = np.empty(N_outer)
    out_n_starts = np.empty(N_outer, dtype=np.int64)

    t0 = time.time()
    for i in range(N_outer):
        # Per-u seed for reproducibility, decorrelated from outer seed
        res = _laplace_is_one_u(
            U[i], V_r, V_perp, callables,
            M_aux=M_aux, n_mode_starts=n_mode_starts,
            seed=seed * 1_000_003 + i,
        )
        out_log_pi_U[i] = res["log_pi_U"]
        out_rel_se[i] = res["rel_se"]
        out_M_eff_frac[i] = res["M_eff_frac"]
        out_log_f[i] = res["log_f_at_mode"]
        out_n_starts[i] = res["n_starts_used"]

        if progress_every is not None and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (N_outer - (i + 1)) / rate
            print(f"  [marginalization] {i+1}/{N_outer} "
                  f"({rate*1000:.1f} u/s, ~{remaining:.0f}s remaining)",
                  flush=True)

    return {
        "log_pi_U": out_log_pi_U,
        "rel_se": out_rel_se,
        "M_eff_frac": out_M_eff_frac,
        "log_f_at_mode": out_log_f,
        "n_starts_used": out_n_starts,
        "wall_time_s": time.time() - t0,
    }


def sample_pi_U(
    N: int, V_r: np.ndarray, benchmark: str = "banana", seed: int = 0,
) -> np.ndarray:
    """Draw N i.i.d. samples from π_U via the noise law.

    Procedure: sample η ~ π_n, rank-Gaussianize to Z (in *population*, this
    is just Z = L W since the marginals are exactly logistic), project U = Z V_r.

    Note: this samples from the *population* π_U, not the empirically
    rank-Gaussianized version that the deployed model uses. The difference
    is a marginal-CDF error of order N^{-1/2} which vanishes for the
    population reference; for Stage-2 verification with high N_pop we want
    the population version, which is what this gives.

    Uses cas.transforms.inv_logit_phi for the eta -> z step (no clipping,
    stable out to |eta| ~ 700).
    """
    from .transforms import inv_logit_phi

    callables = _get_active_callables(benchmark)
    sample_noise_fn = callables["sample_noise"]

    rng = np.random.default_rng(seed)
    eta = sample_noise_fn(N, rng)
    # Rank-Gaussianize via the *population* marginal of the noise law.
    # For banana/cubic_banana the population marginal is the standard
    # logistic; PPG has a different active marginal (handled via inv_logit_phi
    # on the eta values, then per-coord pair-marginal lookup if needed,
    # but since the active z is what we project, inv_logit_phi(eta) gives
    # z directly for any of the three benchmarks).
    Z = inv_logit_phi(eta)
    return Z @ V_r
