"""Stage-1 reduced KL at an arbitrary V_r by nested Monte Carlo,

    KL(pi_Z || pi_Z(.; V_r)) = E_{z ~ pi_Z}[ log c(z) - log c_r(V_r^T z) ],
    c_r(u) = E_{zeta ~ N(0, I - V_r V_r^T)}[ c(V_r u + zeta) ].

The inner estimate of log c_r is biased low at finite M_inner.
Public: stage1_oracle_kl(V_r, log_c_z, sample_z, N_outer, M_inner, rng).
"""
from __future__ import annotations

import numpy as np


def stage1_oracle_kl(
    V_r: np.ndarray,
    log_c_z: callable,
    sample_z: callable,
    N_outer: int = 2000,
    M_inner: int = 4096,
    rng: np.random.Generator | None = None,
    return_per_sample: bool = False,
) -> dict:
    """Stage-1-isolated KL D_KL(pi_Z || pi_Z(. ; V_r)) via nested MC.

    Parameters
    ----------
    V_r : (d, r) ndarray
        Orthonormal columns spanning the rank-r reduced subspace.
    log_c_z : callable
        log c(z): (N, d) -> (N,). The log of the rank-Gaussianized copula
        density (NOT the eta-space pi_eta).
    sample_z : callable
        Sampler: (N, rng) -> (N, d). Draws z from pi_Z (rank-Gaussianized).
    N_outer : int
        Number of outer samples z ~ pi_Z.
    M_inner : int
        Inner-MC sample size for estimating c_r(u).
    rng : np.random.Generator, optional
        RNG. Defaults to a fresh seed.
    return_per_sample : bool
        If True, also include per-sample log c(z) and log c_r-hat(V^T z)
        in the output for diagnostics.

    Returns
    -------
    dict with keys:
        kl_est : float
            MC estimate of D_KL(pi_Z || pi_Z(. ; V_r)) in nats.
        kl_se : float
            Standard error of kl_est (across outer samples).
        ess_min : float
            Minimum effective sample size of the inner IS-like estimator
            across the N_outer outer samples. Small values (<100) indicate
            inner-MC variance is hurting; bump M_inner.
        N_outer, M_inner : int
            Bookkeeping.
        log_c_per_sample, log_cr_per_sample : (N_outer,) ndarray, optional
            Per-outer-sample values (only if return_per_sample=True).
    """
    if rng is None:
        rng = np.random.default_rng()

    V_r = np.asarray(V_r, dtype=np.float64)
    d, r = V_r.shape
    if r > d:
        raise ValueError(f"V_r has r={r} > d={d}")

    # Orthonormality check (defensive — caller may pass non-orthonormal V_r)
    VtV = V_r.T @ V_r
    if not np.allclose(VtV, np.eye(r), atol=1e-8):
        raise ValueError("V_r must have orthonormal columns")

    P_perp = np.eye(d) - V_r @ V_r.T  # projector onto perp complement
    # Cholesky of P_perp (rank d-r; use eigendecomposition for safety)
    eigvals_pp, eigvecs_pp = np.linalg.eigh(P_perp)
    # Numerical: P_perp has rank d-r exactly; eigvals are 0 (r times) and 1 (d-r times)
    pos_mask = eigvals_pp > 1e-10
    sqrt_P_perp = eigvecs_pp[:, pos_mask] * np.sqrt(eigvals_pp[pos_mask])
    # sqrt_P_perp: (d, d-r); samples z_perp ~ N(0, P_perp) = sqrt_P_perp @ N(0, I_{d-r})

    # ---- Outer loop: sample z ~ pi_Z ----
    z = sample_z(N_outer, rng)
    log_c_outer = log_c_z(z)  # (N_outer,)

    # ---- Project z onto V_r ----
    u = z @ V_r            # (N_outer, r)
    z_active = u @ V_r.T   # (N_outer, d) — V_r u for each outer, in one matmul

    # ---- Inner MC: estimate log c_r(u) for each outer sample ----
    # c_r(u) = E_{zeta ~ N(0, P_perp)} [ c(V_r u + zeta) ]
    # Shared inner zeta samples (common random numbers): one draw, reused
    # across outer samples, gives variance reduction at no extra cost.
    xi = rng.standard_normal((M_inner, d - r))      # (M_inner, d-r)
    zeta_inner = xi @ sqrt_P_perp.T                  # (M_inner, d)

    log_cr_outer = np.empty(N_outer)
    ess_per_outer = np.empty(N_outer)

    # Per-outer loop.  Empirically, batching multiple outer samples into one
    # log_c_z call doesn't help (the underlying numpy ops are memory-bound,
    # so larger arrays just push data out of cache without amortizing
    # Python overhead).  Keep one outer sample per call; reuse a scratch
    # buffer to avoid allocating a fresh (M_inner, d) per iteration.
    z_cand = np.empty((M_inner, d))
    for n in range(N_outer):
        # z_cand = z_active[n] + zeta_inner, in-place
        np.add(z_active[n], zeta_inner, out=z_cand)
        log_c_inner = log_c_z(z_cand)               # (M_inner,)
        m_max = log_c_inner.max()
        w = np.exp(log_c_inner - m_max)
        sum_w = w.sum()
        sumsq_w = (w * w).sum()
        log_cr_outer[n] = m_max + np.log(sum_w / M_inner)
        ess_per_outer[n] = (sum_w * sum_w) / sumsq_w if sumsq_w > 0 else 0.0

    # ---- KL estimate ----
    contrib = log_c_outer - log_cr_outer  # (N_outer,)
    kl_est = float(contrib.mean())
    kl_se = float(contrib.std(ddof=1) / np.sqrt(N_outer)) if N_outer > 1 else float("nan")

    out = {
        "kl_est": kl_est,
        "kl_se": kl_se,
        "ess_min": float(ess_per_outer.min()),
        "ess_mean": float(ess_per_outer.mean()),
        "N_outer": int(N_outer),
        "M_inner": int(M_inner),
    }
    if return_per_sample:
        out["log_c_per_sample"] = log_c_outer
        out["log_cr_per_sample"] = log_cr_outer
    return out
