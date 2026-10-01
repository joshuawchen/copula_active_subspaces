"""Joint estimate of pi_U(u) and KL(pi_{W|U=u} || gamma_{d-r}) per u, for the
Stage-1 reference KL by conditional Monte Carlo,

    KL(pi_n || pi_n(.; V_r)) = E_{u ~ pi_U}[ KL(pi_{W|U=u} || gamma_{d-r}) ].

inner_sampler="laplace_is": Laplace-centred multi-start importance sampling.
inner_sampler="gmm": HMC samples, a Gaussian-mixture fit and importance
sampling from the mixture, with Laplace importance sampling as fall-back.
"""
from __future__ import annotations
import time
from typing import Optional

import numpy as np

from .oracle_kl_marginalization import (
    _get_active_callables,
    _multistart_mode,
    _orthonormal_complement,
)

# Acceptance gate for an IS value used as a quadrature fall-back.
#
# The quad path (cas.quad_marginalization) hands back a u whenever its
# reduction does not apply, and the fall-back is _laplace_is_one_u_with_kl --
# the very estimator whose catastrophic outliers the quad path was written to
# remove. Its own diagnostics detect those failures: M_eff_frac is the
# effective sample size fraction of the self-normalized IS weights, and rel_se
# the relative standard error of the log_pi_U estimator. Both were computed and
# then never consulted, so a u whose weights had collapsed onto one draw still
# contributed its inner KL to the cell mean at face value.
#
# These thresholds are deliberately loose: they are a poison filter, not a
# convergence criterion. An IS pass at M_eff_frac >= 1e-3 of M_aux = 1e5 still
# has ~100 effective draws, while the observed failures ran at 1e-5 to 1e-6.
IS_FALLBACK_MIN_ESS_FRAC: float = 1e-3
IS_FALLBACK_MAX_REL_SE: float = 0.10


def _laplace_is_one_u_with_kl(
    u: np.ndarray,
    V_r: np.ndarray,
    V_perp: np.ndarray,
    callables: dict,
    M_aux: int,
    n_mode_starts: int,
    seed: int,
) -> dict:
    """One-u IS estimate of (log π_U(u), D_KL(π_{W|U=u} ‖ γ_{d-r})).

    Identical IS sampling pass to _laplace_is_one_u; just computes the inner
    KL as a self-normalized IS expectation on the same samples.

    Returns:
        dict with:
            log_pi_U:        scalar
            inner_kl:        scalar D_KL(π_{W|U=u} ‖ γ_{d-r})
            inner_kl_se:     self-normalized IS SE of inner_kl
            rel_se:           scalar rel SE of the log_pi_U estimator
            M_eff_frac:       effective sample size fraction
            log_f_at_mode:    log f(w*) at the chosen Laplace center
            n_starts_used:    how many Newton starts before the best was found
            inner_sampler_used: 'laplace_is'
    """
    d = callables["d"]
    k_act = callables["k_act"]
    L_inv = callables["L_inv"]
    log_det_L = callables["log_det_L"]
    log_pi_W_active = callables["log_pi_W_active"]
    grad_log_pi_W_active = callables["grad_log_pi_W_active"]
    hess_log_pi_W_active = callables["hess_log_pi_W_active"]

    dm = V_perp.shape[1]  # d - r

    # Per-u precomputable quantities
    u_tilde = L_inv @ (V_r @ u)
    w_tilde = L_inv @ V_perp
    w_tilde_act = w_tilde[:k_act, :]
    w_tilde_res = w_tilde[k_act:, :]
    u_tilde_act = u_tilde[:k_act]
    u_tilde_res = u_tilde[k_act:]

    # Gaussian-only IS proposal (matches the residual γ factor).
    # When r < k_act (e.g. PPG at r=3, k_act=6), w_tilde_res has shape
    # (d-k_act, dm) with dm > d-k_act and is rank-deficient, so G_res
    # is rank d-k_act < dm and inv(G_res) is numerically degenerate.
    # Fall back to a standard-Gaussian start; Newton finds the mode using
    # the full Hessian.
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

    # Active + residual log-densities at the IS samples
    W_act_eval = u_tilde_act[None, :] + w_samples @ w_tilde_act.T
    log_active = log_pi_W_active(W_act_eval)
    W_res_eval = u_tilde_res[None, :] + w_samples @ w_tilde_res.T
    log_gauss_res = (
        -0.5 * (W_res_eval ** 2).sum(axis=1)
        - 0.5 * (d - k_act) * np.log(2 * np.pi)
    )
    log_integrand = log_active + log_gauss_res
    # log π_Z(V_r u + V_⊥ w) = log π_W(L^{-1}(...)) - log|det L|
    log_pi_Z_samples = log_integrand - log_det_L

    # Proposal density at the IS samples
    diff = w_samples - w_star
    log_q = (
        -0.5 * np.einsum("mi,ij,mj->m", diff, Sigma_q_inv, diff)
        - 0.5 * dm * np.log(2 * np.pi)
        - 0.5 * log_det_Sigma_q
    )

    log_w = log_integrand - log_q
    log_w_max = log_w.max()
    w_norm = np.exp(log_w - log_w_max)

    # log π_U(u) = log|det L|^{-1} * E_q[integrand / q]
    log_pi_U = log_w_max + np.log(w_norm.mean()) - log_det_L

    # IS variance diagnostics for log_pi_U
    M_eff = (w_norm.sum() ** 2) / (w_norm ** 2).sum()
    rel_se = float(w_norm.std(ddof=1) / w_norm.mean() / np.sqrt(M_aux))

    # Self-normalized IS estimator of the inner conditional expectation
    #   E_{w~π_{W|U}}[log π_Z(V_r u + V_⊥ w) - log γ_{d-r}(w)]
    log_gamma_dm = (
        -0.5 * (w_samples ** 2).sum(axis=1) - 0.5 * dm * np.log(2 * np.pi)
    )
    integrand_inner = log_pi_Z_samples - log_gamma_dm     # (M_aux,)

    w_norm_sum = float(w_norm.sum())
    if w_norm_sum <= 0:
        raise RuntimeError(
            "self-normalized IS weights summed to zero; degenerate proposal"
        )
    tilde_w = w_norm / w_norm_sum                         # (M_aux,)
    cond_expect = float((tilde_w * integrand_inner).sum())

    # Self-normalized IS variance (approximation; see Owen 2013, eq 9.7):
    #   Var ≈ Σ_m tilde_w_m^2 (integrand_m - cond_expect)^2
    # This is a standard SNIS variance estimator; it's biased low when
    # M_eff << M_aux but is the right asymptotic.
    var_inner = float(np.sum(tilde_w ** 2 * (integrand_inner - cond_expect) ** 2))

    inner_kl = -log_pi_U + cond_expect
    inner_kl_se = float(np.sqrt(max(var_inner, 0.0)))

    return {
        "log_pi_U": float(log_pi_U),
        "inner_kl": float(inner_kl),
        "inner_kl_se": inner_kl_se,
        "rel_se": rel_se,
        "M_eff_frac": float(M_eff / M_aux),
        "log_f_at_mode": float(log_f_best),
        "n_starts_used": int(n_starts_used),
        "inner_sampler_used": "laplace_is",
    }


# =====================================================================
# GMM-based inner sampler
# =====================================================================
# Newton mode → HMC chain on log_f → GMM fit on HMC samples → IS pass.
#
# This is the "good importance sampler" for benchmarks whose integrand
# geometry defeats bare Laplace IS — specifically banana, where Laplace
# gives M_eff/M_aux ~ 10⁻³ while GMM-IS gives ~5×10⁻² (≈50× better).
#
# Rationale (see also: experiments/diag_banana_gmm.py):
#   * The integrand log f(w) is degree-4 polynomial in w (banana, ppg) or
#     degree-6 (cubic_banana). Its mass concentrates along a curved
#     submanifold that no single Gaussian centred at the mode covers well.
#   * HMC on log f produces samples from π_{W|U=u} directly.
#   * A K-component GMM fit to those samples is rich enough to approximate
#     the curved geometry while remaining a tractable IS proposal (closed-form
#     log q for evaluation, closed-form sampling).
#
# Default K = 16 is empirically near-optimal at HMC sample sizes 5k–20k.

def _make_log_target(u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
                     callables, d, k_act):
    """Build scalar (log_f, grad_log_f) callables for HMC."""
    log_pi_W_active = callables["log_pi_W_active"]
    grad_log_pi_W_active = callables["grad_log_pi_W_active"]

    def log_f(w):
        a = u_tilde_act + w_tilde_act @ w
        b = u_tilde_res + w_tilde_res @ w
        try:
            la = float(log_pi_W_active(a[None, :])[0])
        except Exception:
            return -np.inf
        if not np.isfinite(la):
            return -np.inf
        return (la - 0.5 * float((b ** 2).sum())
                - 0.5 * (d - k_act) * np.log(2 * np.pi))

    def grad_log_f(w):
        a = u_tilde_act + w_tilde_act @ w
        try:
            grad_act = grad_log_pi_W_active(a[None, :])[0]
        except Exception:
            return np.zeros_like(w)
        if not np.all(np.isfinite(grad_act)):
            return np.zeros_like(w)
        return (w_tilde_act.T @ grad_act
                - w_tilde_res.T @ (u_tilde_res + w_tilde_res @ w))

    return log_f, grad_log_f


def _hmc_sample(log_f, grad_log_f, w0, L_mass, n_samples, n_leapfrog=40,
                init_step_size=0.1, n_warmup=1000, target_accept=0.7,
                rng=None):
    """HMC with multiplicative step-size adaptation during warmup.

    L_mass: Cholesky factor of mass matrix M = Σ_q^{-1}; momenta drawn as
    p = L_mass @ randn(dm), giving kinetic energy ½ p^T Σ_q p.

    Returns: (samples, accept_rate, final_step_size).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    dm = w0.shape[0]
    M_inv = np.linalg.solve(L_mass @ L_mass.T, np.eye(dm))
    samples = np.zeros((n_samples, dm))
    w = w0.copy()
    log_f_w = log_f(w)
    grad_w = grad_log_f(w)
    eps = init_step_size
    eps_min, eps_max = 1e-5, 0.5
    accept_history = []
    recent_window = 50
    sample_idx = 0

    for it in range(n_warmup + n_samples):
        p0 = L_mass @ rng.standard_normal(dm)
        K0 = 0.5 * p0 @ M_inv @ p0
        H0 = -log_f_w + K0
        w_new = w.copy()
        p = p0.copy()
        grad_new = grad_w.copy()
        diverged = False
        for _ in range(n_leapfrog):
            p = p + 0.5 * eps * grad_new
            w_new = w_new + eps * (M_inv @ p)
            if not np.all(np.isfinite(w_new)) or np.linalg.norm(w_new) > 1e6:
                diverged = True
                break
            grad_new = grad_log_f(w_new)
            if not np.all(np.isfinite(grad_new)):
                diverged = True
                break
            p = p + 0.5 * eps * grad_new
        if diverged:
            alpha = 0.0
        else:
            log_f_new = log_f(w_new)
            K_new = 0.5 * p @ M_inv @ p
            H_new = -log_f_new + K_new
            if not np.isfinite(H_new):
                alpha = 0.0
            else:
                log_alpha = -(H_new - H0)
                alpha = min(1.0, np.exp(min(log_alpha, 0)))
        if not diverged and rng.uniform() < alpha:
            w = w_new
            log_f_w = log_f_new
            grad_w = grad_new
        accept_history.append(alpha)
        if it < n_warmup and (it + 1) % recent_window == 0:
            recent_accept = np.mean(accept_history[-recent_window:])
            if recent_accept < target_accept - 0.1:
                eps = max(eps_min, eps * 0.8)
            elif recent_accept > target_accept + 0.1:
                eps = min(eps_max, eps * 1.25)
        if it >= n_warmup:
            samples[sample_idx] = w
            sample_idx += 1

    accept_rate = float(np.mean(accept_history[n_warmup:]))
    return samples, accept_rate, eps


def _gmm_is_one_u_with_kl(
    u: np.ndarray,
    V_r: np.ndarray,
    V_perp: np.ndarray,
    callables: dict,
    M_aux: int,
    n_mode_starts: int,
    seed: int,
    n_hmc: int = 5000,
    n_warmup: int = 1500,
    n_leapfrog: int = 40,
    gmm_K: int = 16,
    gmm_reg_covar: float = 1e-6,
    hmc_accept_threshold: float = 0.3,
) -> dict:
    """One-u GMM-IS estimate of (log π_U(u), inner KL).

    Procedure:
        1. Newton mode w_star from Laplace seed.
        2. Σ_q from inverse Hessian at the mode.
        3. HMC on log_f using Σ_q as mass-matrix preconditioner.
        4. Fit a gmm_K-component Gaussian mixture to HMC samples.
        5. Sample M_aux from the GMM; compute IS estimates.

    Falls back to _laplace_is_one_u_with_kl if HMC fails to mix
    (accept_rate < hmc_accept_threshold) or GMM fit fails.

    Returns dict with same keys as _laplace_is_one_u_with_kl plus:
        hmc_accept_rate, hmc_step_size, gmm_K_used, gmm_logL_train
    """
    d = callables["d"]
    k_act = callables["k_act"]
    L_inv = callables["L_inv"]
    log_det_L = callables["log_det_L"]
    log_pi_W_active = callables["log_pi_W_active"]
    grad_log_pi_W_active = callables["grad_log_pi_W_active"]
    hess_log_pi_W_active = callables["hess_log_pi_W_active"]
    dm = V_perp.shape[1]

    u_tilde = L_inv @ (V_r @ u)
    w_tilde = L_inv @ V_perp
    w_tilde_act = w_tilde[:k_act, :]
    w_tilde_res = w_tilde[k_act:, :]
    u_tilde_act = u_tilde[:k_act]
    u_tilde_res = u_tilde[k_act:]

    # Stage A: Newton mode (same as Laplace path)
    rank_full = (w_tilde_res.shape[0] >= dm)
    if rank_full:
        G_res = w_tilde_res.T @ w_tilde_res + 1e-10 * np.eye(dm)
        Sigma_gauss = np.linalg.inv(G_res)
        mu_gauss = -Sigma_gauss @ w_tilde_res.T @ u_tilde_res
    else:
        Sigma_gauss = np.eye(dm)
        mu_gauss = np.zeros(dm)

    w_star, log_f_best, n_starts_used = _multistart_mode(
        u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
        mu_gauss, Sigma_gauss,
        log_pi_W_active, grad_log_pi_W_active, hess_log_pi_W_active,
        n_starts=n_mode_starts, seed=seed,
    )

    # Stage B: Σ_q from inverse Hessian at the mode
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

    # Stage C: HMC chain from w_star using M = Σ_q^{-1}
    log_f, grad_log_f = _make_log_target(
        u_tilde_act, u_tilde_res, w_tilde_act, w_tilde_res,
        callables, d, k_act,
    )
    M_mass = np.linalg.inv(Sigma_q)
    M_mass = 0.5 * (M_mass + M_mass.T)
    eigs_m = np.linalg.eigvalsh(M_mass)
    if eigs_m.min() <= 0:
        M_mass = M_mass + (1e-8 - eigs_m.min()) * np.eye(dm)
    L_mass = np.linalg.cholesky(M_mass)
    rng_hmc = np.random.default_rng(seed + 30000)
    hmc_samples, accept_rate, step_size = _hmc_sample(
        log_f, grad_log_f, w_star, L_mass, n_hmc,
        n_leapfrog=n_leapfrog, init_step_size=0.1,
        n_warmup=n_warmup, target_accept=0.7, rng=rng_hmc,
    )

    # Fallback: HMC didn't mix
    if accept_rate < hmc_accept_threshold:
        res = _laplace_is_one_u_with_kl(
            u, V_r, V_perp, callables, M_aux, n_mode_starts, seed,
        )
        res["inner_sampler_used"] = "laplace_fallback_hmc"
        res["hmc_accept_rate"] = float(accept_rate)
        res["hmc_step_size"] = float(step_size)
        res["gmm_K_used"] = 0
        res["gmm_logL_train"] = float("nan")
        return res

    # Stage D: fit a GMM to HMC samples
    try:
        from sklearn.mixture import GaussianMixture
        # Clip seed to 32-bit range (sklearn requirement)
        gmm_seed = int(seed) % (2 ** 32)
        # Defensive clamp: each GMM component needs at minimum ~dm + 2 samples
        # to fit a regularized covariance. require >= 50 samples per component.
        max_K = max(2, n_hmc // 50)
        K_used = min(gmm_K, max_K)
        gmm = GaussianMixture(
            n_components=K_used,
            covariance_type="full",
            reg_covar=gmm_reg_covar,
            max_iter=200,
            n_init=1,
            random_state=gmm_seed,
        )
        gmm.fit(hmc_samples)
        gmm_logL_train = float(gmm.score(hmc_samples))
    except Exception:
        # GMM fit failed → fall back to Laplace
        res = _laplace_is_one_u_with_kl(
            u, V_r, V_perp, callables, M_aux, n_mode_starts, seed,
        )
        res["inner_sampler_used"] = "laplace_fallback_gmm"
        res["hmc_accept_rate"] = float(accept_rate)
        res["hmc_step_size"] = float(step_size)
        res["gmm_K_used"] = 0
        res["gmm_logL_train"] = float("nan")
        return res

    # Stage E: IS pass with GMM proposal
    rng_is = np.random.default_rng(seed + 40000)
    # Sample M_aux from the GMM. gmm.sample draws using its own RNG seeded
    # at fit time; for reproducibility we pre-seed sklearn via random_state.
    w_samples, _ = gmm.sample(M_aux)
    # Shuffle the GMM-sample order (gmm.sample groups by component) so any
    # downstream order-dependent statistics aren't biased.
    perm = rng_is.permutation(M_aux)
    w_samples = w_samples[perm]

    # Active + residual log-densities at IS samples
    W_act_eval = u_tilde_act[None, :] + w_samples @ w_tilde_act.T
    log_active = log_pi_W_active(W_act_eval)
    W_res_eval = u_tilde_res[None, :] + w_samples @ w_tilde_res.T
    log_gauss_res = (
        -0.5 * (W_res_eval ** 2).sum(axis=1)
        - 0.5 * (d - k_act) * np.log(2 * np.pi)
    )
    log_integrand = log_active + log_gauss_res
    log_pi_Z_samples = log_integrand - log_det_L

    # GMM proposal density at IS samples (closed form via sklearn)
    log_q = gmm.score_samples(w_samples)

    log_w = log_integrand - log_q
    log_w_max = log_w.max()
    w_norm = np.exp(log_w - log_w_max)
    log_pi_U = log_w_max + np.log(w_norm.mean()) - log_det_L

    M_eff = (w_norm.sum() ** 2) / (w_norm ** 2).sum()
    rel_se = float(w_norm.std(ddof=1) / w_norm.mean() / np.sqrt(M_aux))

    # Fallback gate: if GMM-IS itself produced ESS < Laplace's typical floor,
    # the GMM fit was uninformative or the IS samples missed the integrand.
    # Fall back to Laplace IS for this u and flag.
    # Threshold tuned so banana cells with HMC mixing failures fall through;
    # 1e-3 covers the lowest typical Laplace floor on banana (~1.5e-3) without
    # tripping legitimate moderate-ESS GMM-IS draws.
    if (M_eff / M_aux) < 1e-3:
        res = _laplace_is_one_u_with_kl(
            u, V_r, V_perp, callables, M_aux, n_mode_starts, seed,
        )
        res["inner_sampler_used"] = "laplace_fallback_lowess"
        res["hmc_accept_rate"] = float(accept_rate)
        res["hmc_step_size"] = float(step_size)
        res["gmm_K_used"] = K_used
        res["gmm_logL_train"] = float(gmm_logL_train)
        return res

    log_gamma_dm = (
        -0.5 * (w_samples ** 2).sum(axis=1) - 0.5 * dm * np.log(2 * np.pi)
    )
    integrand_inner = log_pi_Z_samples - log_gamma_dm

    w_norm_sum = float(w_norm.sum())
    if w_norm_sum <= 0:
        # Numerical degeneracy → fall back
        res = _laplace_is_one_u_with_kl(
            u, V_r, V_perp, callables, M_aux, n_mode_starts, seed,
        )
        res["inner_sampler_used"] = "laplace_fallback_isweights"
        res["hmc_accept_rate"] = float(accept_rate)
        res["hmc_step_size"] = float(step_size)
        res["gmm_K_used"] = K_used
        res["gmm_logL_train"] = float(gmm_logL_train)
        return res
    tilde_w = w_norm / w_norm_sum
    cond_expect = float((tilde_w * integrand_inner).sum())
    var_inner = float(np.sum(tilde_w ** 2 * (integrand_inner - cond_expect) ** 2))
    inner_kl = -log_pi_U + cond_expect
    inner_kl_se = float(np.sqrt(max(var_inner, 0.0)))

    return {
        "log_pi_U": float(log_pi_U),
        "inner_kl": float(inner_kl),
        "inner_kl_se": inner_kl_se,
        "rel_se": rel_se,
        "M_eff_frac": float(M_eff / M_aux),
        "log_f_at_mode": float(log_f_best),
        "n_starts_used": int(n_starts_used),
        "inner_sampler_used": "gmm",
        "hmc_accept_rate": float(accept_rate),
        "hmc_step_size": float(step_size),
        "gmm_K_used": int(K_used),
        "gmm_logL_train": float(gmm_logL_train),
    }


def marginalize_pi_U_with_inner_kl(
    U: np.ndarray,
    V_r: np.ndarray,
    benchmark: str = "banana",
    *,
    M_aux: int = 100_000,
    n_mode_starts: int = 20,
    seed: int = 0,
    progress_every: Optional[int] = None,
    inner_sampler: str = "laplace_is",
    n_hmc: int = 5000,
    n_warmup: int = 1500,
    n_leapfrog: int = 40,
    gmm_K: int = 16,
    nq: int = 14,
) -> dict:
    """Compute (log π_U(u), D_KL(π_{W|U=u} ‖ γ_{d-r})) for each u in U.

    Args:
        U: (N_outer, r) — query points in projected coords.
        V_r: (d, r) — orthonormal subspace.
        benchmark: noise-law key.
        M_aux: IS sample count per u.
        n_mode_starts: Newton multi-start count.
        seed: RNG seed; reproducible.
        progress_every: print a progress line every N u's.
        inner_sampler: "laplace_is" (default; fast, low memory, ~1s/u on
            banana but with ESS ~ 10⁻³) or "gmm" (HMC + GMM-IS; ~5s/u on
            banana but with ESS ~ 5×10⁻²).
        n_hmc, n_warmup, n_leapfrog, gmm_K: GMM-sampler knobs (ignored when
            inner_sampler == "laplace_is").

    Returns:
        dict with per-u arrays. With inner_sampler="gmm" the following
        extra arrays are present: hmc_accept_rate, hmc_step_size,
        gmm_K_used, gmm_logL_train; plus a per-u string array
        inner_sampler_used recording which sampler was used (and any
        fallback to Laplace).
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    N_outer, r = U.shape

    V_r = np.asarray(V_r, dtype=float)
    if V_r.shape[1] != r:
        raise ValueError(
            f"V_r has r={V_r.shape[1]} but U has r={r}; they must match."
        )

    if inner_sampler not in ("laplace_is", "gmm", "quad"):
        raise ValueError(
            f"inner_sampler must be 'laplace_is', 'gmm', or 'quad', got {inner_sampler!r}"
        )

    callables = _get_active_callables(benchmark)
    d = callables["d"]
    if V_r.shape[0] != d:
        raise ValueError(
            f"V_r has d={V_r.shape[0]} but benchmark '{benchmark}' has d={d}"
        )

    V_perp = _orthonormal_complement(V_r)

    if inner_sampler == "quad":
        # Deterministic quadrature: drops the Gaussian residual directions and
        # Gauss-Hermites the k_act-dim active block (cas.quad_marginalization).
        from .quad_marginalization import marginalize_pi_U_with_inner_kl_quad
        q = marginalize_pi_U_with_inner_kl_quad(
            U, V_r, benchmark, nq=nq, n_mode_starts=n_mode_starts,
            seed=seed, progress_every=progress_every,
        )
        # IS fall-back for the u's the quadrature reduction could not handle.
        # The fall-back is GATED, not blind: _laplace_is_one_u_with_kl is the
        # estimator whose catastrophic-outlier failure mode this module exists
        # to remove, so an IS value is accepted only when its own diagnostics
        # say it converged. A rejected u is left NaN and counted; the caller
        # aggregates over valid u's and fails loudly when too many are lost.
        # Before this gate, every rejected u entered the cell mean at face
        # value, which is how single u's of order 1e7 nats reached the cache.
        for i in np.where(q["inner_sampler_used"] == "quad_skip")[0]:
            reason = str(q["quad_skip_reason"][i]) if "quad_skip_reason" in q \
                else "unknown"
            res = _laplace_is_one_u_with_kl(
                U[i], V_r, V_perp, callables, M_aux=M_aux,
                n_mode_starts=n_mode_starts, seed=seed * 1_000_003 + i,
            )
            ok = (np.isfinite(res["inner_kl"])
                  and res["M_eff_frac"] >= IS_FALLBACK_MIN_ESS_FRAC
                  and res["rel_se"] <= IS_FALLBACK_MAX_REL_SE)
            if ok:
                for _k in ("log_pi_U", "inner_kl", "inner_kl_se",
                           "rel_se", "M_eff_frac"):
                    q[_k][i] = res[_k]
                q["inner_sampler_used"][i] = f"laplace_fallback_{reason}"
            else:
                for _k in ("log_pi_U", "inner_kl", "inner_kl_se"):
                    q[_k][i] = np.nan
                q["rel_se"][i] = res["rel_se"]
                q["M_eff_frac"][i] = res["M_eff_frac"]
                q["inner_sampler_used"][i] = f"rejected_{reason}"
        return q

    out_log_pi_U = np.empty(N_outer)
    out_inner_kl = np.empty(N_outer)
    out_inner_kl_se = np.empty(N_outer)
    out_rel_se = np.empty(N_outer)
    out_M_eff = np.empty(N_outer)
    out_log_f = np.empty(N_outer)
    out_n_starts = np.empty(N_outer, dtype=np.int64)
    out_sampler_used = []
    out_hmc_accept = np.full(N_outer, np.nan)
    out_hmc_step = np.full(N_outer, np.nan)
    out_gmm_K = np.zeros(N_outer, dtype=np.int64)

    t0 = time.time()
    for i in range(N_outer):
        if inner_sampler == "laplace_is":
            res = _laplace_is_one_u_with_kl(
                U[i], V_r, V_perp, callables,
                M_aux=M_aux, n_mode_starts=n_mode_starts,
                seed=seed * 1_000_003 + i,
            )
        else:  # "gmm"
            res = _gmm_is_one_u_with_kl(
                U[i], V_r, V_perp, callables,
                M_aux=M_aux, n_mode_starts=n_mode_starts,
                seed=seed * 1_000_003 + i,
                n_hmc=n_hmc, n_warmup=n_warmup, n_leapfrog=n_leapfrog,
                gmm_K=gmm_K,
            )
        out_log_pi_U[i] = res["log_pi_U"]
        out_inner_kl[i] = res["inner_kl"]
        out_inner_kl_se[i] = res["inner_kl_se"]
        out_rel_se[i] = res["rel_se"]
        out_M_eff[i] = res["M_eff_frac"]
        out_log_f[i] = res["log_f_at_mode"]
        out_n_starts[i] = res["n_starts_used"]
        out_sampler_used.append(res.get("inner_sampler_used", inner_sampler))
        if "hmc_accept_rate" in res:
            out_hmc_accept[i] = res["hmc_accept_rate"]
        if "hmc_step_size" in res:
            out_hmc_step[i] = res["hmc_step_size"]
        if "gmm_K_used" in res:
            out_gmm_K[i] = res["gmm_K_used"]

        if progress_every is not None and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (N_outer - (i + 1)) / rate
            print(
                f"  [marginalization+KL] {i+1}/{N_outer} "
                f"({rate:.1f} u/s, ~{remaining:.0f}s remaining)",
                flush=True,
            )

    return {
        "log_pi_U": out_log_pi_U,
        "inner_kl": out_inner_kl,
        "inner_kl_se": out_inner_kl_se,
        "rel_se": out_rel_se,
        "M_eff_frac": out_M_eff,
        "log_f_at_mode": out_log_f,
        "n_starts_used": out_n_starts,
        "inner_sampler_used": np.array(out_sampler_used),
        "hmc_accept_rate": out_hmc_accept,
        "hmc_step_size": out_hmc_step,
        "gmm_K_used": out_gmm_K,
        "wall_time_s": time.time() - t0,
    }
