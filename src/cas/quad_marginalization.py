"""Gauss-Hermite version of the conditional Monte Carlo estimator of
cas.conditional_kl: log pi_U(u) and KL(pi_{W|U=u} || gamma_{d-r}) by
Laplace-centred, multi-mode Gauss-Hermite quadrature over the k_act active
latent coordinates. Requires r >= k_act; returns None where the reduction
does not apply, for the caller's fall-back.
"""
from __future__ import annotations
import time
from typing import Optional

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import logsumexp

from .oracle_kl_marginalization import (
    _get_active_callables,
    _orthonormal_complement,
)

_LOG2PI = float(np.log(2.0 * np.pi))
_GH_CACHE: dict = {}


def _gh_tensor(nq: int, k: int):
    """Tensor Gauss-Hermite nodes/weights for E_{N(0,I_k)}[f]. Cached."""
    key = (nq, k)
    if key not in _GH_CACHE:
        x, w = hermegauss(nq)
        w = w / np.sqrt(2.0 * np.pi)
        grids = np.meshgrid(*([x] * k), indexing="ij")
        wgrids = np.meshgrid(*([w] * k), indexing="ij")
        Xi = np.stack([g.ravel() for g in grids], axis=1)
        Wq = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
        _GH_CACHE[key] = (Xi, Wq)
    return _GH_CACHE[key]


def _newton_one(s0, mu_s, S_s_inv, u_act, grad_act, hess_act, k):
    """Single Newton ascent of  -0.5 (s-mu_s)^T S_s_inv (s-mu_s) + log_active(u_act+s).
    Returns (s, ok)."""
    s = np.asarray(s0, dtype=float).copy()
    for _ in range(50):
        a = (u_act + s)[None, :]
        g = -S_s_inv @ (s - mu_s) + grad_act(a)[0]
        H = -S_s_inv + hess_act(a)[0]
        H = 0.5 * (H + H.T)
        eig = np.linalg.eigvalsh(H)
        if eig.max() > -1e-8:
            H = H - (eig.max() + 1e-3) * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return s, False
        s = s - step
        if np.linalg.norm(step) < 1e-9:
            break
    return s, np.all(np.isfinite(s))


def _find_modes(mu_s, S_s_inv, u_act, callables, n_starts, seed):
    """Multi-start Newton; return a list of (s_star, negH) for every DISTINCT
    local maximum (positive-definite negative Hessian). Multi-modality can occur
    for the multi-to-one conformal maps at far bases; at reference bases there is
    a single mode and this returns one entry.
    """
    log_act = callables["log_pi_W_active"]
    grad_act = callables["grad_log_pi_W_active"]
    hess_act = callables["hess_log_pi_W_active"]
    k = mu_s.shape[0]
    rng = np.random.default_rng(seed)

    def obj(s):
        ds = s - mu_s
        return -0.5 * ds @ S_s_inv @ ds + float(log_act((u_act + s)[None, :])[0])

    try:
        scale = np.sqrt(np.clip(np.diag(np.linalg.inv(S_s_inv)), 1e-12, None))
    except np.linalg.LinAlgError:
        scale = np.ones(k)

    starts = [mu_s.copy(), np.zeros(k), -u_act.copy()]
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(mu_s + scale * rng.standard_normal(k) * 2.0)

    found = []          # list of (s, obj_val)
    for s0 in starts:
        s, ok = _newton_one(s0, mu_s, S_s_inv, u_act, grad_act, hess_act, k)
        if not ok:
            continue
        H = -S_s_inv + hess_act((u_act + s)[None, :])[0]
        H = 0.5 * (H + H.T)
        if np.linalg.eigvalsh(H).max() > -1e-8:
            continue    # not a strict local max
        if all(np.linalg.norm(s - sp) > 1e-3 for sp, _ in found):
            found.append((s, obj(s)))

    if not found:
        raise RuntimeError("s-space Newton found no local maximum")

    out = []
    for s, _ in found:
        negH = S_s_inv - hess_act((u_act + s)[None, :])[0]
        negH = 0.5 * (negH + negH.T)
        eigp = np.linalg.eigvalsh(negH)
        if eigp.min() <= 0:
            negH = negH + (1e-8 - eigp.min()) * np.eye(k)
        out.append((s, negH))
    return out


def _quad_one_u_with_kl(u, V_r, V_perp, callables, nq, n_mode_starts, seed):
    """Multi-mode Laplace-GH estimate of (log pi_U(u), inner KL).

    On success returns the per-u result dict. When the quadrature reduction is
    not applicable, returns {"skip_reason": <str>} instead, and the caller
    decides what to do with that u.  The reason MATTERS and must not be
    collapsed to a bare None: 'ss_collapse' is benign (the complement carries
    no active tilt, so the inner KL is ~0 and the IS fall-back returns it
    correctly), whereas 'no_mode' means the tilted integrand had no strict
    local maximum at this u, which is where the IS fall-back's catastrophic
    outliers come from."""
    d = callables["d"]
    k_act = callables["k_act"]
    L_inv = callables["L_inv"]
    log_det_L = callables["log_det_L"]
    log_act = callables["log_pi_W_active"]
    dm = V_perp.shape[1]
    r = V_r.shape[1]

    u_tilde = L_inv @ (V_r @ u)
    w_tilde = L_inv @ V_perp
    A = w_tilde[:k_act, :]
    B = w_tilde[k_act:, :]
    u_act = u_tilde[:k_act]
    u_res = u_tilde[k_act:]

    if (d - k_act) < dm:
        return {"skip_reason": "rank_deficient"}
    BtB = B.T @ B
    cond = np.linalg.cond(BtB)
    if not np.isfinite(cond) or cond > 1e12:
        return {"skip_reason": "btb_ill_conditioned"}

    Sig_w = np.linalg.inv(BtB)
    m_w = -Sig_w @ (B.T @ u_res)
    mu_s = A @ m_w
    S_s = A @ Sig_w @ A.T
    S_s = 0.5 * (S_s + S_s.T)
    # Guard the reduction against a degenerate active projection. When V_r
    # captures (nearly) all active directions, A = w_tilde_act -> 0, so S_s
    # collapses in SCALE (not condition number: cond can stay O(1) while
    # eig.max() -> 0). There the complement carries no active tilt, the
    # reduction is exact-Gaussian, and the s-integral is ill-posed; hand such
    # cells to the IS fall-back (it returns the correct ~0 inner-KL there).
    eig_Ss = np.linalg.eigvalsh(S_s)
    if eig_Ss.max() < 1e-8 or eig_Ss.min() <= 1e-12 * eig_Ss.max():
        return {"skip_reason": "ss_collapse"}
    S_s_inv = np.linalg.inv(S_s)
    _, logdet_BtB = np.linalg.slogdet(BtB)
    _, logdet_Ss = np.linalg.slogdet(S_s)
    q0 = float(u_res @ u_res - m_w @ BtB @ m_w)

    cross = Sig_w @ A.T @ S_s_inv
    Sig_c = Sig_w - cross @ A @ Sig_w
    trSc = float(np.trace(Sig_c))
    trBScB = float(np.trace(B @ Sig_c @ B.T))

    # _find_modes raises when multi-start Newton locates no strict local
    # maximum, which happens at far bases (large sin Theta) where the tilted
    # s-integrand is flat or its curvature is indefinite. That is a property of
    # this one u, not of the run, so hand the point to the IS fall-back rather
    # than aborting: the None return is the same contract used above for the
    # rank-deficient and degenerate-projection cases.
    try:
        modes = _find_modes(mu_s, S_s_inv, u_act, callables, n_mode_starts, seed)
    except RuntimeError:
        return {"skip_reason": "no_mode"}

    # mixture-proposal components: (s_k, negH_k, logdet_negH_k, log-Laplace-mass_k)
    comps = []
    for s_k, negH_k in modes:
        _, logdet_negH = np.linalg.slogdet(negH_k)
        ds = s_k - mu_s
        logphi = (-0.5 * ds @ S_s_inv @ ds
                  + float(log_act((u_act + s_k)[None, :])[0]))
        log_mass = logphi - 0.5 * logdet_negH          # + 0.5 log|Sig_k|
        comps.append((s_k, negH_k, logdet_negH, log_mass))
    lm = np.array([c[3] for c in comps])
    log_pi_k = lm - logsumexp(lm)                      # mixture log-weights

    Xi, Wq = _gh_tensor(nq, k_act)
    logWq = np.log(Wq)
    all_S = []
    all_logterm = []
    for (s_k, negH_k, logdet_negH, _), lpk in zip(comps, log_pi_k):
        Lc = np.linalg.cholesky(np.linalg.inv(negH_k)
                                + 1e-14 * np.eye(k_act))
        S = s_k[None, :] + Xi @ Lc.T
        dsb = S - mu_s[None, :]
        logN_base = (-0.5 * np.einsum("mi,ij,mj->m", dsb, S_s_inv, dsb)
                     - 0.5 * logdet_Ss - 0.5 * k_act * _LOG2PI)
        # mixture density log q_mix(S) = logsumexp_j [log pi_j + logN(S;s_j,negH_j)]
        lq = []
        for (s_j, negH_j, logdet_negH_j, _), lpj in zip(comps, log_pi_k):
            dj = S - s_j[None, :]
            lq.append(lpj - 0.5 * np.einsum("mi,ij,mj->m", dj, negH_j, dj)
                      + 0.5 * logdet_negH_j - 0.5 * k_act * _LOG2PI)
        log_qmix = logsumexp(np.array(lq), axis=0)
        la = log_act(u_act[None, :] + S)
        logterm = lpk + logWq + logN_base + la - log_qmix
        all_S.append(S)
        all_logterm.append(logterm)
    S = np.concatenate(all_S, axis=0)
    logterm = np.concatenate(all_logterm, axis=0)
    logEt = logsumexp(logterm)

    log_pi_U = (-log_det_L + 0.5 * (k_act - r) * _LOG2PI
                - 0.5 * logdet_BtB - 0.5 * q0 + logEt)

    wt = np.exp(logterm - logterm.max())
    wt = wt / wt.sum()
    la_all = log_act(u_act[None, :] + S)
    MC = m_w[None, :] + (S - mu_s[None, :]) @ cross.T
    E_wsq = float((wt * ((MC ** 2).sum(1) + trSc)).sum())
    resmean = u_res[None, :] + MC @ B.T
    E_ressq = float((wt * ((resmean ** 2).sum(1) + trBScB)).sum())
    E_active = float((wt * la_all).sum())

    E_log_piZ = (E_active - 0.5 * E_ressq
                 - 0.5 * (d - k_act) * _LOG2PI - log_det_L)
    E_log_gamma = -0.5 * E_wsq - 0.5 * dm * _LOG2PI
    inner_kl = E_log_piZ - log_pi_U - E_log_gamma

    ess = 1.0 / float((wt ** 2).sum())
    return {
        "log_pi_U": float(log_pi_U),
        "inner_kl": float(inner_kl),
        "inner_kl_se": 0.0,
        "rel_se": 0.0,
        "M_eff_frac": float(ess / len(wt)),
        "log_f_at_mode": float("nan"),
        "n_starts_used": int(len(modes)),
        "inner_sampler_used": "quad",
    }


def marginalize_pi_U_with_inner_kl_quad(
    U: np.ndarray,
    V_r: np.ndarray,
    benchmark: str = "banana",
    *,
    nq: int = 14,
    n_mode_starts: int = 12,
    seed: int = 0,
    progress_every: Optional[int] = None,
) -> dict:
    """Deterministic quadrature analogue of marginalize_pi_U_with_inner_kl.
    Same return dict shape (per-u arrays); rank-deficient cells are marked
    'quad_skip' in inner_sampler_used (caller should fall back to IS)."""
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    N_outer, r = U.shape
    V_r = np.asarray(V_r, dtype=float)
    if V_r.shape[1] != r:
        raise ValueError(f"V_r has r={V_r.shape[1]} but U has r={r}")
    callables = _get_active_callables(benchmark)
    d = callables["d"]
    if V_r.shape[0] != d:
        raise ValueError(f"V_r has d={V_r.shape[0]} but benchmark has d={d}")
    V_perp = _orthonormal_complement(V_r)

    keys = ("log_pi_U", "inner_kl", "inner_kl_se", "rel_se", "M_eff_frac")
    out = {k: np.empty(N_outer) for k in keys}
    # Named n_starts_used to match the IS estimator's return dict, which the
    # docstring promises and which exp_stage1_reference consumes; the
    # n_modes alias below is kept for older readers of this cache.
    out["n_starts_used"] = np.empty(N_outer, dtype=np.int64)
    used = []
    skip_reasons = []
    t0 = time.time()
    for i in range(N_outer):
        res = _quad_one_u_with_kl(
            U[i], V_r, V_perp, callables,
            nq=nq, n_mode_starts=n_mode_starts, seed=seed * 1_000_003 + i,
        )
        if res is None or "skip_reason" in res:
            for k in keys:
                out[k][i] = np.nan
            out["n_starts_used"][i] = 0
            used.append("quad_skip")
            skip_reasons.append("unknown" if res is None
                                else str(res["skip_reason"]))
        else:
            for k in keys:
                out[k][i] = res[k]
            out["n_starts_used"][i] = res["n_starts_used"]
            used.append("quad")
            skip_reasons.append("")
        if progress_every is not None and (i + 1) % progress_every == 0:
            print(f"  [quad marg+KL] {i+1}/{N_outer} "
                  f"({(i+1)/(time.time()-t0):.1f} u/s)", flush=True)
    out["n_modes"] = out["n_starts_used"]
    out["inner_sampler_used"] = np.array(used)
    out["quad_skip_reason"] = np.array(skip_reasons)
    out["wall_time_s"] = time.time() - t0
    return out
