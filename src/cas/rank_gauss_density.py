"""Log-density of the rank-Gaussianized noise from a closed-form eta-space
density and empirical marginals,

    log pi_Z(z) = log pi_eta(eta(z)) + sum_j [ log phi(z_j) - log f_j(eta_j) ],

and compute_population_MHC, which returns M, H and C (C as T T^T / N and as
H + M - 2I).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

from .reduced_density import ScalarMarginal, rank_transform_through_marginals


_LOG_2PI = float(np.log(2.0 * np.pi))


def make_log_pi_Z(
    log_pi_eta_fn: Callable[[np.ndarray], np.ndarray],
    marginals: Sequence[ScalarMarginal],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the log-density log pi_Z(z) for a benchmark.

    Parameters
    ----------
    log_pi_eta_fn : callable
        Closed-form log pi_eta(eta) for the benchmark, taking
        eta of shape (N, d) and returning (N,).
    marginals : sequence of ScalarMarginal, length d
        Per-coord empirical marginals of eta. These supply both
        the inverse CDF (z -> eta) and log f_eta_j (Jacobian piece).

    Returns
    -------
    log_pi_Z : callable
        log_pi_Z(z) for z of shape (N, d), returning (N,).
        Uses log_pi_eta(eta(z)) + sum_j [log phi(z_j) - log f_j(eta_j(z_j))].
    """
    d_mar = len(marginals)

    def log_pi_Z(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.ndim != 2:
            raise ValueError(f"log_pi_Z expects z of shape (N, d), got {z.shape}")
        N, d = z.shape
        if d != d_mar:
            raise ValueError(
                f"log_pi_Z: z has d={d} but marginals has length {d_mar}"
            )
        # eta_j = F_j^{-1}(Phi(z_j))
        # ScalarMarginal.inverse_cdf takes a u in (0,1).
        u = norm.cdf(z)
        eta = np.empty_like(z)
        for j in range(d):
            eta[:, j] = marginals[j].inverse_cdf(u[:, j])
        # log pi_eta(eta)
        log_pe = log_pi_eta_fn(eta)
        # Jacobian: sum_j [log phi(z_j) - log f_j(eta_j)]
        log_phi_z = -0.5 * (z * z).sum(axis=-1) - 0.5 * d * _LOG_2PI
        log_f_eta = np.zeros(N)
        for j in range(d):
            log_f_eta += marginals[j].log_pdf(eta[:, j])
        return log_pe + log_phi_z - log_f_eta

    return log_pi_Z


def fit_marginals_from_eta(eta: np.ndarray) -> list:
    """Fit per-coord ScalarMarginal from an eta sample.

    Convenience wrapper that returns a list of d ScalarMarginals,
    one per column of eta. Used together with make_log_pi_Z to set
    up a population-level analysis from a large eta sample.

    Parameters
    ----------
    eta : (N, d) array of samples.

    Returns
    -------
    marginals : list of d ScalarMarginal.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2:
        raise ValueError(f"eta must be (N, d), got {eta.shape}")
    N, d = eta.shape
    return [ScalarMarginal(eta[:, j]) for j in range(d)]


def rank_gauss_with_marginals(eta: np.ndarray, marginals: Sequence[ScalarMarginal]
                               ) -> np.ndarray:
    """Apply the per-coord rank-Gauss z = Phi^{-1}(F_j(eta_j)) using
    a fixed set of ScalarMarginal objects.

    Thin wrapper around rank_transform_through_marginals for symmetry
    with the rest of this module. Use this when you want to project a
    test sample of eta onto a previously-fit set of marginals.
    """
    return rank_transform_through_marginals(np.asarray(eta, dtype=float), marginals)


def compute_population_MHC(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    log_pi_eta_fn: Callable[[np.ndarray], np.ndarray],
    N: int,
    seed: int = 0,
    h_fd: float = 1e-3,
) -> dict:
    """Compute population (M, H, C_direct, C_stein) for one benchmark.

    Returns a dict with:
      z              (N, d) array, rank-Gauss'd samples
      score          (N, d) array, grad log pi_Z at z via central FD
      M, H           (d, d), the second-moment and Fisher matrices
      C_direct       (d, d) = T T^T / N with T = score + z
      C_stein        (d, d) = H + M - 2 I (Stein-identity-implied C)
      stein_residual_F   ||C_direct - C_stein||_F (smaller is closer to
                          the population limit)
      ibp_residual_F     ||(score^T @ z) / N + I||_F
      marginals      list of ScalarMarginal used for rank-Gauss

    At the population limit C_direct == C_stein, but at finite N they
    differ by MC noise. Use C_stein for the bound-dominance comparison
    (J_hi <= 0.5 tr((I-VV')C_stein) is an algebraic identity); use
    C_direct as a sanity check on Stein convergence (the residual
    quantifies how close N is to the population limit).
    """
    rng = np.random.default_rng(seed)
    eta = sampler(N, rng)
    d = eta.shape[1]
    marginals = fit_marginals_from_eta(eta)
    log_pi_Z = make_log_pi_Z(log_pi_eta_fn, marginals)
    z = rank_gauss_with_marginals(eta, marginals)
    score = np.empty_like(z)
    for j in range(d):
        zp = z.copy(); zp[:, j] += h_fd
        zm = z.copy(); zm[:, j] -= h_fd
        score[:, j] = (log_pi_Z(zp) - log_pi_Z(zm)) / (2 * h_fd)
    T_pipe = score + z
    M = (z.T @ z) / N
    H = (score.T @ score) / N
    C_direct = (T_pipe.T @ T_pipe) / N
    C_stein = H + M - 2 * np.eye(d)
    ibp = (score.T @ z) / N
    return {
        "z": z,
        "score": score,
        "M": M,
        "H": H,
        "C_direct": C_direct,
        "C_stein": C_stein,
        "stein_residual_F": float(np.linalg.norm(C_direct - C_stein, "fro")),
        "ibp_residual_F": float(np.linalg.norm(ibp + np.eye(d), "fro")),
        "marginals": marginals,
        "N": N,
        "h_fd": h_fd,
    }
