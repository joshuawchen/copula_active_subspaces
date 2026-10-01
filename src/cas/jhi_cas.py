"""Basis minimizing the J_hi bound on the Stage-1 KL (Part II supplement), by
Riemannian gradient descent on the Grassmann manifold from the CAS basis,
with (M, H, C) estimated by MHC_from_T_memmap.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


# Tiny ridge on H_perp for inversion robustness. Population H is positive
# definite under regularity; finite-sample H_hat may be near-singular along
# directions of low score variance, in which case the gradient is dominated
# by the M term anyway, so a tiny ridge does not change the geometry.
_H_RIDGE_DEFAULT = 1e-10


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def orthonormal_complement(V_r: np.ndarray) -> np.ndarray:
    """(d, d-r) orthonormal complement of (d, r) basis V_r.

    Returns columns spanning the orthogonal complement of V_r.
    """
    d, r = V_r.shape
    Q, _ = np.linalg.qr(np.concatenate([V_r, np.eye(d)], axis=1))
    return Q[:, r:d]


def sin_theta_F(V1: np.ndarray, V2: np.ndarray) -> float:
    """Frobenius sin-Theta distance between two r-dim subspaces of R^d.

    Both V1 and V2 should be orthonormal (d x r). Returns
        sqrt(sum_i 1 - sigma_i^2)
    where sigma_i are singular values of V1.T @ V2.
    """
    s = np.linalg.svd(V1.T @ V2, compute_uv=False)
    s = np.clip(s, 0, 1)
    return float(np.sqrt(np.sum(1.0 - s**2)))


# ---------------------------------------------------------------------------
# Numerically safe inverse for H_perp
# ---------------------------------------------------------------------------

def _safe_invert(A: np.ndarray, ridge: float = _H_RIDGE_DEFAULT) -> np.ndarray:
    """Symmetric-PSD inversion with a tiny ridge for stability.

    Used for H_perp throughout. Falls back to pseudoinverse if the matrix is
    deeply singular (eigenvalues below 100*ridge).
    """
    n = A.shape[0]
    A_sym = 0.5 * (A + A.T)
    # Try direct ridge-augmented solve first.
    try:
        return np.linalg.solve(A_sym + ridge * np.eye(n), np.eye(n))
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse via eigendecomposition.
        w, V = np.linalg.eigh(A_sym)
        w_safe = np.where(w > 100 * ridge, 1.0 / w, 0.0)
        return V @ np.diag(w_safe) @ V.T


def _safe_slogdet(A: np.ndarray) -> Optional[float]:
    """Return log|det(A)| for symmetric PSD A; None if A is not positive."""
    A_sym = 0.5 * (A + A.T)
    eig = np.linalg.eigvalsh(A_sym)
    if eig.min() <= 1e-12:
        return None
    return float(np.sum(np.log(eig)))


# ---------------------------------------------------------------------------
# J_hi value and gradient
# ---------------------------------------------------------------------------

def J_hi(V_r: np.ndarray, M: np.ndarray, H: np.ndarray) -> Optional[float]:
    """LCLMZ J_hi upper bound on Stage-1 KL at orthonormal V_r.

        J_hi(V_r) = 0.5 (tr(V_perp^T M V_perp) - (d - r) + log det(V_perp^T H V_perp))

    Returns None if V_perp^T H V_perp is not positive definite (numerical guard).
    """
    d, r = V_r.shape
    V_perp = orthonormal_complement(V_r)
    M_perp = V_perp.T @ M @ V_perp
    H_perp = V_perp.T @ H @ V_perp
    ldH = _safe_slogdet(H_perp)
    if ldH is None:
        return None
    return 0.5 * (float(np.trace(M_perp)) - (d - r) + ldH)


def J_hi_grad_ambient(V_r: np.ndarray, M: np.ndarray, H: np.ndarray,
                      ridge: float = _H_RIDGE_DEFAULT) -> np.ndarray:
    """Closed-form Riemannian gradient (Proposition 2), ambient frame.

    Returns (d x r) ambient gradient = V_perp * [-V_perp^T M V_r - H_perp^{-1} V_perp^T H V_r].

    The gradient lies in the horizontal-tangent space (column space of V_perp).
    Uses a numerically safe inverse of H_perp (ridge-regularized).
    """
    V_perp = orthonormal_complement(V_r)
    H_perp = V_perp.T @ H @ V_perp
    H_perp_inv = _safe_invert(H_perp, ridge=ridge)
    Delta_M = V_perp.T @ M @ V_r
    Delta_H = V_perp.T @ H @ V_r
    G_B = -(Delta_M + H_perp_inv @ Delta_H)
    return V_perp @ G_B


def J_hi_grad_B(V_r: np.ndarray, M: np.ndarray, H: np.ndarray,
                ridge: float = _H_RIDGE_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form gradient in B-coords plus V_perp.

    Returns (G_B, V_perp) where G_B has shape (d-r, r) and is the gradient
    component along V_perp's columns.
    """
    V_perp = orthonormal_complement(V_r)
    H_perp = V_perp.T @ H @ V_perp
    H_perp_inv = _safe_invert(H_perp, ridge=ridge)
    Delta_M = V_perp.T @ M @ V_r
    Delta_H = V_perp.T @ H @ V_r
    G_B = -(Delta_M + H_perp_inv @ Delta_H)
    return G_B, V_perp


# ---------------------------------------------------------------------------
# Closed-form Hessian (Proposition 3) — not used by default but available
# ---------------------------------------------------------------------------

def J_hi_Q_value(B: np.ndarray, V_r: np.ndarray, M: np.ndarray, H: np.ndarray,
                 ridge: float = _H_RIDGE_DEFAULT) -> float:
    """Quadratic coefficient Q(B) in J_hi(V_r(tB)) = J_hi(V_r) + t<G,B> + t^2 Q(B) + O(t^3).

    Closed form (verified to ~1e-6 relative error vs finite differences):
        Q(B) = 0.5 tr(M_act B^T B) - 0.5 tr(M_perp B B^T)
             + 0.5 tr(H_perp^{-1} B H_act B^T) - 0.5 ||B||_F^2
             - 0.25 tr((H_perp^{-1} K_1)^2)
    with K_1 = -(B Delta_H^T + Delta_H B^T) and Delta_H = V_perp^T H V_r.
    """
    V_perp = orthonormal_complement(V_r)
    M_act = V_r.T @ M @ V_r
    H_act = V_r.T @ H @ V_r
    M_perp = V_perp.T @ M @ V_perp
    H_perp = V_perp.T @ H @ V_perp
    H_perp_inv = _safe_invert(H_perp, ridge=ridge)
    Delta_H = V_perp.T @ H @ V_r

    K1 = -(B @ Delta_H.T + Delta_H @ B.T)
    HK1 = H_perp_inv @ K1

    return (
        0.5 * float(np.trace(M_act @ (B.T @ B)))
        - 0.5 * float(np.trace(M_perp @ (B @ B.T)))
        + 0.5 * float(np.trace(H_perp_inv @ B @ H_act @ B.T))
        - 0.5 * float(np.linalg.norm(B) ** 2)
        - 0.25 * float(np.trace(HK1 @ HK1))
    )


# ---------------------------------------------------------------------------
# Trace CAS baseline
# ---------------------------------------------------------------------------

def trace_cas_basis(C: np.ndarray, r: int) -> np.ndarray:
    """Trace CAS: top-r eigvecs of C.

    Args:
        C: (d, d) tempered-score covariance E[T T^T]. Should be PSD.
        r: subspace rank.

    Returns:
        V_r: (d, r) orthonormal basis (top-r eigvecs of C).
    """
    d = C.shape[0]
    if r > d:
        raise ValueError(f"r ({r}) must be <= d ({d})")
    eig, V_full = np.linalg.eigh(C)  # ascending
    return V_full[:, ::-1][:, :r]    # descending, take top-r


# ---------------------------------------------------------------------------
# J_hi CAS via Grassmann gradient descent
# ---------------------------------------------------------------------------

def jhi_grassmann_descent(
    M: np.ndarray,
    H: np.ndarray,
    r: int,
    V_init: Optional[np.ndarray] = None,
    n_iter: int = 500,
    tol: float = 1e-9,
    step_grid: tuple = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5),
    ridge: float = _H_RIDGE_DEFAULT,
) -> Tuple[np.ndarray, dict]:
    """J_hi CAS basis via Riemannian gradient descent on the Grassmann manifold.

    Args:
        M: (d, d) second-moment matrix E[Z Z^T]. Should be PSD.
        H: (d, d) Fisher info matrix E[score score^T]. Should be PSD.
        r: subspace rank.
        V_init: (d, r) initial basis (orthonormal). If None, defaults to trace CAS
            warm start using C = H + M - 2I (Stein identity).
        n_iter: max iterations.
        tol: gradient norm tolerance for convergence.
        step_grid: backtracking line-search step sizes (descending).
        ridge: small ridge on H_perp for inversion robustness.

    Returns:
        V_opt: (d, r) orthonormal J_hi minimizer.
        info: dict with n_iters, final_J_hi, final_grad_norm, terminated_for.

    Algorithm:
        1. Start from V_init (or warm-start at trace CAS).
        2. Compute closed-form gradient G in ambient frame.
        3. Backtracking line search along -G, retract via QR.
        4. Repeat until ||G||_F < tol or no step improves J_hi.

    Properties:
        - Converges in 20-50 iterations from CAS warm start (5-15 ms typical).
        - No local minima observed empirically; multiple random inits converge
          to the same V_opt.
        - Numerically safe: H_perp inversion uses a tiny ridge; J_hi guards
          for non-PD H_perp. Loop terminates if J_hi becomes non-evaluable.
    """
    d = M.shape[0]
    if M.shape != (d, d) or H.shape != (d, d):
        raise ValueError(f"M and H must be (d, d) of same size; got {M.shape}, {H.shape}")
    if V_init is None:
        # Default: trace CAS warm start. Use the algebraic identity C = H + M - 2I.
        C_alg = H + M - 2 * np.eye(d)
        V = trace_cas_basis(C_alg, r)
    else:
        if V_init.shape != (d, r):
            raise ValueError(f"V_init must be ({d}, {r}); got {V_init.shape}")
        # Re-orthogonalize for safety.
        Q, _ = np.linalg.qr(V_init)
        V = Q[:, :r]

    last_J = J_hi(V, M, H)
    iters_used = 0
    terminated_for = "max_iters"
    for it in range(n_iter):
        iters_used = it + 1
        G = J_hi_grad_ambient(V, M, H, ridge=ridge)
        norm_g = float(np.linalg.norm(G))
        if norm_g < tol:
            terminated_for = "tolerance"
            break
        f = J_hi(V, M, H)
        if f is None:
            # H_perp degenerated; bail out with last good V.
            terminated_for = "H_perp_non_pd"
            break
        # Backtracking line search
        best_step = 0.0
        for step in step_grid:
            V_try = V - step * G
            Q, _ = np.linalg.qr(V_try)
            V_try = Q[:, :r]
            f_try = J_hi(V_try, M, H)
            if f_try is not None and f_try < f - 1e-15:
                best_step = step
                break
        if best_step == 0.0:
            terminated_for = "no_descent_step"
            break
        V_new = V - best_step * G
        Q, _ = np.linalg.qr(V_new)
        V = Q[:, :r]
        last_J = J_hi(V, M, H)

    return V, {
        "n_iters": iters_used,
        "final_J_hi": float(last_J) if last_J is not None else float("nan"),
        "final_grad_norm": float(np.linalg.norm(J_hi_grad_ambient(V, M, H, ridge=ridge))),
        "terminated_for": terminated_for,
    }


# ---------------------------------------------------------------------------
# Extraction of (M, H, C) from a Hermite fit
# ---------------------------------------------------------------------------

def MHC_from_T_memmap(
    Z_path: str,
    T_path: str,
    N: int,
    d: int,
    N_chunk: int = 80000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (M, H, C) from saved Z and T memmaps produced by Hermite fit.

    The Hermite fit (cas.stage1_reference.stream_T_at_basis) writes T = score
    estimator (tempered) to T_path. The rank-Gaussianized Z is at Z_path.

    Computes:
        M_hat = (1/N) Z^T Z          : score-free second moment.
        H_hat = (1/N) (T-Z)^T (T-Z)  : Fisher info from estimated score s = T - z.
        C_hat = (1/N) T^T T          : tempered-score covariance.

    Returns:
        M_hat, H_hat, C_hat: each (d, d) float64, PSD by construction.

    Note: by Stein's lemma, E[C] = E[H] + E[M] - 2I. The three estimators
    do not satisfy this exactly at finite N (they differ by MC noise) but
    converge to the identity as N -> infinity.
    """
    Z = np.load(Z_path, mmap_mode='r')
    T = np.load(T_path, mmap_mode='r')
    if Z.shape[0] != N or Z.shape[1] != d:
        raise ValueError(f"Z shape {Z.shape} mismatch with (N, d) = ({N}, {d})")
    if T.shape != (d, N):
        raise ValueError(f"T shape {T.shape} mismatch with (d, N) = ({d}, {N})")

    M_hat = np.zeros((d, d), dtype=np.float64)
    H_hat = np.zeros((d, d), dtype=np.float64)
    C_hat = np.zeros((d, d), dtype=np.float64)
    for start in range(0, N, N_chunk):
        end = min(start + N_chunk, N)
        Zc = np.asarray(Z[start:end], dtype=np.float64)        # (Nc, d)
        Tc = np.asarray(T[:, start:end], dtype=np.float64).T   # (Nc, d)
        sc = Tc - Zc
        M_hat += Zc.T @ Zc
        H_hat += sc.T @ sc
        C_hat += Tc.T @ Tc
    M_hat /= N
    H_hat /= N
    C_hat /= N
    return M_hat, H_hat, C_hat


# ---------------------------------------------------------------------------
# Bound evaluation
# ---------------------------------------------------------------------------

def evaluate_bounds(V_r: np.ndarray, M: np.ndarray, H: np.ndarray, C: np.ndarray,
                    z_mean: Optional[np.ndarray] = None) -> dict:
    """Evaluate LCLMZ bounds at V_r using (M, H, C) (typically reference / oracle).

    Returns dict with:
        J_hi, J_lo, thm41_rhs : the standard bounds.
        tr_M_perp, tr_H_perp, tr_C_perp : block traces in V_perp.
        logdet_H_perp, logdet_C_cov_perp : log-determinants used internally.

    All quantities are deterministic functions of V_r and (M, H, C[, z_mean]).
    Use a high-precision (M, H, C) estimate (e.g., from N=1M Hermite fit) to
    get clean bound values that aren't dominated by in-sample MC noise.
    """
    d, r = V_r.shape
    V_perp = orthonormal_complement(V_r)
    out = {}

    M_perp = V_perp.T @ M @ V_perp
    H_perp = V_perp.T @ H @ V_perp
    C_perp = V_perp.T @ C @ V_perp
    out["tr_M_perp"] = float(np.trace(M_perp))
    out["tr_H_perp"] = float(np.trace(H_perp))
    out["tr_C_perp"] = float(np.trace(C_perp))

    # J_hi (using safe slogdet)
    ldH = _safe_slogdet(H_perp)
    if ldH is not None:
        out["J_hi"] = 0.5 * (out["tr_M_perp"] - (d - r) + ldH)
        out["logdet_H_perp"] = ldH
    else:
        out["J_hi"] = None
        out["J_hi_warning"] = "H_perp not PD"

    # J_lo (uses M - z_mean z_mean^T = covariance)
    if z_mean is None:
        z_mean = np.zeros(d)
    Cov = M - z_mean[:, None] @ z_mean[None, :]
    Cov_perp = V_perp.T @ Cov @ V_perp
    ldC = _safe_slogdet(Cov_perp)
    if ldC is not None:
        out["J_lo"] = 0.5 * (out["tr_M_perp"] - (d - r) - ldC)
        out["logdet_C_cov_perp"] = ldC
    else:
        out["J_lo"] = None

    # Thm 4.1 RHS = 0.5 tr(V_perp^T C V_perp)
    out["thm41_rhs"] = 0.5 * out["tr_C_perp"]

    return out
