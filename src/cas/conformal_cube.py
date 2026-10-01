"""Example 3 of Part I. The outputs are the saturated real and imaginary parts
of (W0 + i W1)^3,

    W2 = c tanh((W0^3 - 3 W0 W1^2) / M) + sqrt(1 - a^2) eps2,
    W3 = c tanh((3 W0^2 W1 - W1^3) / M) + sqrt(1 - a^2) eps3,

with M = 4 and a = 0.7, mixed by the rotation of cas.mixing and mapped by
eta = logit(Phi(z)).

Public: sample_conformal_cube_noise, log_conformal_cube_density,
oracle_subspace_conformal_cube, D_C3, R_C3_TARGET, ALPHA_C3, M_C3.
"""
from __future__ import annotations
import numpy as np

from .mixing import block_rotation, B_SUPPORT
from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
D_C3: int = 20            # ambient dimension
R_C3_TARGET: int = 4      # active subspace rank (2 inputs + 2 outputs)
ALPHA_C3: float = 0.7     # conditional-mean variance fraction (Var(mu) = a^2)
M_C3: float = 4.0         # saturation scale (tanh knee of the z^3 argument)
#
# ALPHA_C3 stays at 0.7.  Under the orthogonal mixing this law's fitted
# tr C_Lambda is 0.658 against 1.61 / 1.63 for Examples 1 and 2, and the obvious
# response -- raise the dependence strength -- is wrong.  Var(mu) = a^2 by
# construction, so the z^3 block already carries the SAME dependence strength as
# z^2; the smaller trace is a property of C_Lambda, which is a FITTED object, and
# the cubic score is harder to represent at the deployed (K,q) = (4,2).  Raising
# a to 0.9 tightens the conditional variance from 0.51 to 0.19, makes the density
# sharper still, and measurably breaks the example: the noise-KL separation from
# the Gaussian copula falls to 1.31x (from 2.29x on Example 1) and at some x* the
# reduced model is WORSE than the Gaussian copula on the posterior.  The low trace
# is the representation floor this example exists to exhibit, not a defect.


# Latent mixing: ORTHOGONAL, supported on the first B_SUPPORT coordinates (see
# cas.mixing).  Was the Cholesky factor of a Toeplitz correlation, whose
# Sigma != I put a full-rank linear term into the copula score.  The z^3 block
# has zero pairwise correlation and unit variances, so Cov(W) = I and therefore
# Sigma = Q Cov(W) Q^T = I exactly.
L_CHOL_C3 = block_rotation(D_C3, B_SUPPORT)


def _c3_g2(W0, W1):
    """tanh(Re(z^3)/M) = tanh((W0^3 - 3 W0 W1^2)/M)."""
    return np.tanh((W0 ** 3 - 3.0 * W0 * W1 * W1) / M_C3)


def _c3_g3(W0, W1):
    """tanh(Im(z^3)/M) = tanh((3 W0^2 W1 - W1^3)/M)."""
    return np.tanh((3.0 * W0 * W0 * W1 - W1 ** 3) / M_C3)


def _c3_amplitude(gf, n: int = 129) -> float:
    """c = a / sqrt(Var_{W0,W1~N(0,1)}[gf]) via 2-D Gauss-Hermite (E[gf] = 0)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    a = x[:, None]; b = x[None, :]
    nrm = 1.0 / np.sqrt(2.0 * np.pi)
    wt = (w[:, None] * w[None, :]) * (nrm ** 2)
    v = gf(a, b)
    mean = float((wt * v).sum())
    var = float((wt * v * v).sum()) - mean * mean
    return float(ALPHA_C3 / np.sqrt(var))


_C3_C2: float = _c3_amplitude(_c3_g2)   # = 1.476 at M=4, a=0.7
_C3_C3: float = _c3_amplitude(_c3_g3)   # = _C3_C2 by symmetry


def _c3_means(W0, W1):
    """Conditional means (mu2, mu3) = (c2 tanh(Re z^3 / M), c3 tanh(Im z^3 / M))."""
    return _C3_C2 * _c3_g2(W0, W1), _C3_C3 * _c3_g3(W0, W1)


def _c3_mu_parts(W0, W1):
    """mu2, mu3 with their first and second W-derivatives (FD-verified).

    u2 = (W0^3 - 3 W0 W1^2)/M,  u3 = (3 W0^2 W1 - W1^3)/M.
    du and d2u from the cubic arguments; mu = c tanh(u);
    d mu_i = c sech^2(u) u_i ; d2 mu_ij = c (-2 t sech^2 u_i u_j + sech^2 u_ij).
    """
    M = M_C3
    u2 = (W0 ** 3 - 3.0 * W0 * W1 * W1) / M
    u3 = (3.0 * W0 * W0 * W1 - W1 ** 3) / M
    u2_0 = 3.0 * (W0 * W0 - W1 * W1) / M; u2_1 = -6.0 * W0 * W1 / M
    u3_0 = 6.0 * W0 * W1 / M;             u3_1 = 3.0 * (W0 * W0 - W1 * W1) / M
    u2_00 = 6.0 * W0 / M; u2_11 = -6.0 * W0 / M; u2_01 = -6.0 * W1 / M
    u3_00 = 6.0 * W1 / M; u3_11 = -6.0 * W1 / M; u3_01 = 6.0 * W0 / M
    t2 = np.tanh(u2); t3 = np.tanh(u3)
    sh2 = 1.0 - t2 * t2; sh3 = 1.0 - t3 * t3
    mu2 = _C3_C2 * t2; mu3 = _C3_C3 * t3
    d2_0 = _C3_C2 * sh2 * u2_0; d2_1 = _C3_C2 * sh2 * u2_1
    d3_0 = _C3_C3 * sh3 * u3_0; d3_1 = _C3_C3 * sh3 * u3_1
    a00 = _C3_C2 * (-2.0 * t2 * sh2 * u2_0 * u2_0 + sh2 * u2_00)
    a11 = _C3_C2 * (-2.0 * t2 * sh2 * u2_1 * u2_1 + sh2 * u2_11)
    a01 = _C3_C2 * (-2.0 * t2 * sh2 * u2_0 * u2_1 + sh2 * u2_01)
    b00 = _C3_C3 * (-2.0 * t3 * sh3 * u3_0 * u3_0 + sh3 * u3_00)
    b11 = _C3_C3 * (-2.0 * t3 * sh3 * u3_1 * u3_1 + sh3 * u3_11)
    b01 = _C3_C3 * (-2.0 * t3 * sh3 * u3_0 * u3_1 + sh3 * u3_01)
    return mu2, mu3, d2_0, d2_1, d3_0, d3_1, a00, a11, a01, b00, b11, b01


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_conformal_cube_W(N, rng):
    """Latent W: conformal z^3 block at coords 0..3 (inputs W0,W1 at 0,1;
    outputs W2,W3 at 2,3), then i.i.d. N(0,1) padding."""
    W = np.empty((N, D_C3))
    s = np.sqrt(1.0 - ALPHA_C3 ** 2)
    W0 = rng.standard_normal(N)
    W1 = rng.standard_normal(N)
    mu2, mu3 = _c3_means(W0, W1)
    W[:, 0] = W0
    W[:, 1] = W1
    W[:, 2] = mu2 + s * rng.standard_normal(N)
    W[:, 3] = mu3 + s * rng.standard_normal(N)
    W[:, 4:] = rng.standard_normal((N, D_C3 - 4))
    return W


def sample_conformal_cube_noise(N, rng):
    """Draw N i.i.d. eta from the conformal z^3 noise law."""
    W = sample_conformal_cube_W(N, rng)
    Zlat = W @ L_CHOL_C3.T
    return logit_phi(Zlat)


# Precomputed constants
_C3_LOG_2PI = float(np.log(2.0 * np.pi))
_C3_S2 = 1.0 - ALPHA_C3 ** 2
_C3_LOG_S2 = float(np.log(_C3_S2))
_C3_LOG_DET_L = float(np.linalg.slogdet(L_CHOL_C3)[1])   # = 0 for an orthogonal Q


# ---------------------------------------------------------------------------
# Single source of truth for the active conformal z^3 block (W-space).
# W_act: (N, 4) = [W0, W1, W2, W3]; W0,W1 ~ N(0,1), W2,W3 | . ~ N(mu, s2).
# Shared with cas.oracle_kl_marginalization via the dispatch.
# ---------------------------------------------------------------------------

def log_pi_W_active(W_act):
    """log pi for the conformal z^3 active block (no Gaussian residual, no
    Jacobian, no Toeplitz mixing). W_act: (N, 4) = [W0, W1, W2, W3]."""
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    mu2, mu3 = _c3_means(W0, W1)
    return (
        -0.5 * (W0 * W0 + W1 * W1)
        - 0.5 * (W2 - mu2) ** 2 / _C3_S2
        - 0.5 * (W3 - mu3) ** 2 / _C3_S2
        - _C3_LOG_S2
        - 2.0 * _C3_LOG_2PI
    )


def grad_log_pi_W_active(W_act):
    """grad log pi_W_active wrt W_act, returns (N, 4)."""
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    mu2, mu3, d2_0, d2_1, d3_0, d3_1 = _c3_mu_parts(W0, W1)[:6]
    r2 = (W2 - mu2) / _C3_S2
    r3 = (W3 - mu3) / _C3_S2
    grad = np.empty_like(W_act)
    grad[:, 0] = -W0 + r2 * d2_0 + r3 * d3_0
    grad[:, 1] = -W1 + r2 * d2_1 + r3 * d3_1
    grad[:, 2] = -r2
    grad[:, 3] = -r3
    return grad


def hess_log_pi_W_active(W_act):
    """Hessian of log pi_W_active wrt W_act, returns (N, 4, 4).

    Input-input block (0,1) carries the conformal curvature; input-output
    cross terms are dmu_k/dW_i / s2; output-output block is -I/s2.
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    N = W_act.shape[0]
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    (mu2, mu3, d2_0, d2_1, d3_0, d3_1,
     a00, a11, a01, b00, b11, b01) = _c3_mu_parts(W0, W1)
    r2 = (W2 - mu2) / _C3_S2
    r3 = (W3 - mu3) / _C3_S2
    H = np.zeros((N, 4, 4))
    H[:, 0, 0] = -1.0 - d2_0 ** 2 / _C3_S2 + r2 * a00 - d3_0 ** 2 / _C3_S2 + r3 * b00
    H[:, 1, 1] = -1.0 - d2_1 ** 2 / _C3_S2 + r2 * a11 - d3_1 ** 2 / _C3_S2 + r3 * b11
    H[:, 0, 1] = H[:, 1, 0] = (
        -d2_1 * d2_0 / _C3_S2 + r2 * a01 - d3_1 * d3_0 / _C3_S2 + r3 * b01
    )
    H[:, 2, 2] = -1.0 / _C3_S2
    H[:, 3, 3] = -1.0 / _C3_S2
    H[:, 0, 2] = H[:, 2, 0] = d2_0 / _C3_S2
    H[:, 1, 2] = H[:, 2, 1] = d2_1 / _C3_S2
    H[:, 0, 3] = H[:, 3, 0] = d3_0 / _C3_S2
    H[:, 1, 3] = H[:, 3, 1] = d3_1 / _C3_S2
    return H


# ---------------------------------------------------------------------------
# Full density, copula density, and oracle subspaces
# ---------------------------------------------------------------------------

def log_conformal_cube_density(eta):
    """Closed-form log pi_eta(eta) for the conformal z^3 law."""
    eta = np.asarray(eta, dtype=float)
    flat = (eta.ndim == 1)
    if flat:
        eta = eta[None, :]
    Zlat = inv_logit_phi(eta)
    W = np.linalg.solve(L_CHOL_C3, Zlat.T).T
    log_pi_block = log_pi_W_active(W[:, :4])
    W_rest = W[:, 4:]
    n_rest = D_C3 - 4
    log_pi_rest = (
        -0.5 * (W_rest * W_rest).sum(axis=1)
        - 0.5 * n_rest * _C3_LOG_2PI
    )
    log_pi_W = log_pi_block + log_pi_rest
    log_jac = logit_phi_log_jacobian_z(Zlat).sum(axis=1)
    out = log_pi_W - _C3_LOG_DET_L + log_jac
    return out[0] if flat else out


def oracle_subspace_geom_conformal_cube(r):
    """Geometric active subspace: orthonormalized span of first r columns of L."""
    V = L_CHOL_C3[:, :r].copy()
    Q, _ = np.linalg.qr(V)
    return Q


def _log_copula_density_conformal_cube(z):
    """log c(z) with the standard-logistic marginal approximation (same as
    Example 1; the exact subspace uses the analytic oracle)."""
    eta = logit_phi(z)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_conformal_cube_density(eta) - log_f_mar.sum(axis=-1)


_C3_VKL_N_REF: int = 30_000
_C3_VKL_H_FD: float = 1e-4
_C3_VKL_SEED: int = 20260513
_C3_VKL_R_MAX: int = 8


def _compute_V_KL_conformal_cube(
    r_max: int = _C3_VKL_R_MAX, N_ref: int = _C3_VKL_N_REF,
    h_fd: float = _C3_VKL_H_FD, seed: int = _C3_VKL_SEED,
) -> np.ndarray:
    """Top-r_max eigvecs of the population score covariance C (conformal z^3),
    via central FD on the smooth analytical density. Cached to disk."""
    import os
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    cache_file = os.path.join(
        cache_dir,
        f"conformal_cube_V_KL_top{r_max}_Nref{N_ref}_h{h_fd:.0e}_seed{seed}.npz",
    )
    if os.path.exists(cache_file):
        return np.load(cache_file)["V_KL"]

    rng = np.random.default_rng(seed)
    z = inv_logit_phi(sample_conformal_cube_noise(N_ref, rng))
    d = z.shape[1]
    T = np.empty_like(z)
    for j in range(d):
        zp = z.copy(); zp[:, j] += h_fd
        zm = z.copy(); zm[:, j] -= h_fd
        T[:, j] = (_log_copula_density_conformal_cube(zp) - _log_copula_density_conformal_cube(zm)) / (2.0 * h_fd)
    C = T.T @ T / N_ref
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(-eigvals)
    V_KL = eigvecs[:, order][:, :r_max]
    if os.path.isdir(cache_dir):
        np.savez(cache_file, V_KL=V_KL, eigvals_top=eigvals[order][:r_max],
                 trC=float(np.trace(C)), r_max=r_max, N_ref=N_ref, h_fd=h_fd, seed=seed)
    return V_KL


_V_KL_C3 = None


def oracle_subspace_conformal_cube(r):
    """Reference active subspace: the exact copula active subspace V_r^C for the
    conformal z^3 law, via cas.deployment_oracle (top-r eigenspace of the exact
    copula-score covariance from cas.analytic_oracle). This is the §5.1 / SM4
    recovery target and Stage-1-isolated-KL floor, distinct from the (K=4, q=2)
    chosen-estimator spectrum. The logistic-marginal FD V_KL is retained as
    _compute_V_KL_conformal_cube for diagnostics only."""
    from .deployment_oracle import deployment_oracle_subspace
    return deployment_oracle_subspace("conformal_cube", r)
