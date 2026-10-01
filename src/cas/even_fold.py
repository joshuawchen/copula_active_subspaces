"""Example 2 of Part I. Two pairs in which each output is an even function of
its input,

    W_{j+1} = c (tanh(gamma (W_j^2 - 1)) - m0) + sqrt(1 - a^2) eps,   j in {0, 2},

with gamma = 0.5 and a = 0.7, mixed by the rotation of cas.mixing and mapped
by eta = logit(Phi(z)).

Public: sample_even_fold_noise, log_even_fold_density,
oracle_subspace_even_fold, D_EF, R_EF_TARGET, ALPHA_EF, GAMMA_EF.
"""
from __future__ import annotations
import numpy as np

from .mixing import block_rotation, B_SUPPORT
from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
D_EF: int = 20            # ambient dimension
N_PAIRS_EF: int = 2       # fold pairs at coords (0,1) and (2,3)
R_EF_TARGET: int = 2 * N_PAIRS_EF   # = 4, active subspace rank
ALPHA_EF: float = 0.7     # conditional-mean variance fraction (Var(mu) = a^2)
GAMMA_EF: float = 0.5     # fold sharpness (tanh knee of the even argument)


# Latent mixing: ORTHOGONAL, supported on the first B_SUPPORT coordinates (see
# cas.mixing).  Was the Cholesky factor of a Toeplitz correlation, whose
# Sigma != I put a full-rank linear term into the copula score.  Cov(W) = I here
# (the fold mean is even against an odd input, so E[W1 mu(W1)] = 0, and every
# coordinate is variance-normalized), so Sigma = Q Cov(W) Q^T = I exactly.
L_CHOL_EF = block_rotation(D_EF, B_SUPPORT)


# ---------------------------------------------------------------------------
# Even fold bend b(W1) = tanh(gamma (W1^2 - 1)) and its first two derivatives.
# u = gamma (W1^2 - 1);  du/dW1 = 2 gamma W1.
#   b   = tanh(u)
#   b'  = sech^2(u) * 2 gamma W1
#   b'' = 2 gamma sech^2(u) (1 - 4 gamma W1^2 tanh(u))
# ---------------------------------------------------------------------------

def _ef_b(W1):
    return np.tanh(GAMMA_EF * (W1 * W1 - 1.0))


def _ef_b_d1(W1):
    t = np.tanh(GAMMA_EF * (W1 * W1 - 1.0))
    return (1.0 - t * t) * (2.0 * GAMMA_EF * W1)


def _ef_b_d2(W1):
    u = GAMMA_EF * (W1 * W1 - 1.0)
    t = np.tanh(u)
    sh = 1.0 - t * t                        # sech^2(u)
    return 2.0 * GAMMA_EF * sh * (1.0 - 4.0 * GAMMA_EF * W1 * W1 * t)


def _ef_normalization(n: int = 129):
    """Center m0 = E[b(W1)] and amplitude c = a / sqrt(Var(b)) via 1-D
    Gauss-Hermite quadrature (the even bend has nonzero mean, so centering
    is required for E[mu] = 0)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    nrm = 1.0 / np.sqrt(2.0 * np.pi)
    b = _ef_b(x)
    m0 = float((w * b).sum() * nrm)
    var = float((w * b * b).sum() * nrm) - m0 * m0
    c = float(ALPHA_EF / np.sqrt(var))
    return m0, c


_EF_M0, _EF_C = _ef_normalization()   # centering and amplitude


def _ef_mu(W1):
    """Conditional mean mu(W1) = c (b(W1) - m0)."""
    return _EF_C * (_ef_b(W1) - _EF_M0)


def _ef_mu_d1(W1):
    """mu'(W1) = c b'(W1)."""
    return _EF_C * _ef_b_d1(W1)


def _ef_mu_d2(W1):
    """mu''(W1) = c b''(W1)."""
    return _EF_C * _ef_b_d2(W1)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_even_fold_W(N, rng):
    """Latent W: N_PAIRS_EF fold pairs at coords 0..3 (input at even col,
    output at odd col), then i.i.d. N(0,1) padding."""
    W = np.empty((N, D_EF))
    s = np.sqrt(1.0 - ALPHA_EF ** 2)
    for j in range(N_PAIRS_EF):
        W1 = rng.standard_normal(N)
        eps = rng.standard_normal(N)
        W[:, 2 * j] = W1
        W[:, 2 * j + 1] = _ef_mu(W1) + s * eps
    W[:, 2 * N_PAIRS_EF:] = rng.standard_normal((N, D_EF - 2 * N_PAIRS_EF))
    return W


def sample_even_fold_noise(N, rng):
    """Draw N i.i.d. eta from the even-fold noise law."""
    W = sample_even_fold_W(N, rng)
    Zlat = W @ L_CHOL_EF.T
    return logit_phi(Zlat)


# Precomputed constants
_EF_LOG_2PI = float(np.log(2.0 * np.pi))
_EF_S2 = 1.0 - ALPHA_EF ** 2
_EF_LOG_S2 = float(np.log(_EF_S2))
_EF_LOG_DET_L = float(np.linalg.slogdet(L_CHOL_EF)[1])   # = 0 for an orthogonal Q


# ---------------------------------------------------------------------------
# Single source of truth for the active fold-pair block (W-space).
# Inputs at even cols, outputs at odd cols; W1 ~ N(0,1), W2|W1 ~ N(mu, s2).
# Shared with cas.oracle_kl_marginalization via the dispatch.
# ---------------------------------------------------------------------------

def log_pi_W_active(W_act):
    """log pi for the N_PAIRS_EF active fold pairs (no Gaussian residual, no
    Jacobian, no Toeplitz mixing). W_act: (N, 2*N_PAIRS_EF), input/output at
    columns (2j, 2j+1)."""
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    W1 = W_act[:, 0:2 * N_PAIRS_EF:2]
    W2 = W_act[:, 1:2 * N_PAIRS_EF:2]
    resid = W2 - _ef_mu(W1)
    return (
        -0.5 * (W1 * W1).sum(axis=1)
        - 0.5 * (resid * resid).sum(axis=1) / _EF_S2
        - 0.5 * N_PAIRS_EF * _EF_LOG_S2
        - N_PAIRS_EF * _EF_LOG_2PI
    )


def grad_log_pi_W_active(W_act):
    """grad log pi_W_active wrt W_act, returns (N, 2*N_PAIRS_EF).

        d/dW1 = -W1 + mu'(W1) (W2 - mu) / s2
        d/dW2 = -(W2 - mu) / s2
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    grad = np.empty_like(W_act)
    W1 = W_act[:, 0:2 * N_PAIRS_EF:2]
    W2 = W_act[:, 1:2 * N_PAIRS_EF:2]
    resid = W2 - _ef_mu(W1)
    grad[:, 0:2 * N_PAIRS_EF:2] = -W1 + _ef_mu_d1(W1) * resid / _EF_S2
    grad[:, 1:2 * N_PAIRS_EF:2] = -resid / _EF_S2
    return grad


def hess_log_pi_W_active(W_act):
    """Hessian of log pi_W_active wrt W_act, returns (N, k, k), k=2*N_PAIRS_EF.

        d2/dW1^2   = -1 + mu''(W1) (W2 - mu) / s2 - mu'(W1)^2 / s2
        d2/dW2^2   = -1 / s2
        d2/dW1 dW2 = mu'(W1) / s2
    Block-diagonal across pairs.
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    N = W_act.shape[0]
    k_act = 2 * N_PAIRS_EF
    H = np.zeros((N, k_act, k_act))
    for j in range(N_PAIRS_EF):
        W1 = W_act[:, 2 * j]
        W2 = W_act[:, 2 * j + 1]
        resid = W2 - _ef_mu(W1)
        d1 = _ef_mu_d1(W1)
        d2 = _ef_mu_d2(W1)
        H[:, 2 * j, 2 * j] = -1.0 + d2 * resid / _EF_S2 - d1 ** 2 / _EF_S2
        H[:, 2 * j + 1, 2 * j + 1] = -1.0 / _EF_S2
        H[:, 2 * j, 2 * j + 1] = d1 / _EF_S2
        H[:, 2 * j + 1, 2 * j] = d1 / _EF_S2
    return H


# ---------------------------------------------------------------------------
# Full density, copula density, and oracle subspaces
# ---------------------------------------------------------------------------

def log_even_fold_density(eta):
    """Closed-form log pi_eta(eta) for the even-fold law.

    eta -> Zlat via inverse logit (stable), Zlat -> W via L^{-1} Zlat,
    log pi_eta = log pi_W(W) - log|det L| + sum_i log|d Zlat_i / d eta_i|.
    """
    eta = np.asarray(eta, dtype=float)
    flat = (eta.ndim == 1)
    if flat:
        eta = eta[None, :]
    Zlat = inv_logit_phi(eta)
    W = np.linalg.solve(L_CHOL_EF, Zlat.T).T

    log_pi_pairs = log_pi_W_active(W[:, :2 * N_PAIRS_EF])
    W_rest = W[:, 2 * N_PAIRS_EF:]
    n_rest = D_EF - 2 * N_PAIRS_EF
    log_pi_rest = (
        -0.5 * (W_rest * W_rest).sum(axis=1)
        - 0.5 * n_rest * _EF_LOG_2PI
    )
    log_pi_W = log_pi_pairs + log_pi_rest
    log_jac = logit_phi_log_jacobian_z(Zlat).sum(axis=1)
    out = log_pi_W - _EF_LOG_DET_L + log_jac
    return out[0] if flat else out


def oracle_subspace_geom_even_fold(r):
    """Geometric active subspace: orthonormalized span of first r columns of L."""
    V = L_CHOL_EF[:, :r].copy()
    Q, _ = np.linalg.qr(V)
    return Q


def _log_copula_density_even_fold(z):
    """log c(z) = log pi_eta(eta(z)) - sum_j log f^mar_j(eta_j(z)) with the
    standard-logistic marginal approximation (same inherited approximation as
    Example 1; the exact subspace uses the analytic oracle)."""
    eta = logit_phi(z)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_even_fold_density(eta) - log_f_mar.sum(axis=-1)


_EF_VKL_N_REF: int = 30_000
_EF_VKL_H_FD: float = 1e-4
_EF_VKL_SEED: int = 20260513
_EF_VKL_R_MAX: int = 8


def _compute_V_KL_even_fold(
    r_max: int = _EF_VKL_R_MAX, N_ref: int = _EF_VKL_N_REF,
    h_fd: float = _EF_VKL_H_FD, seed: int = _EF_VKL_SEED,
) -> np.ndarray:
    """Top-r_max eigvecs of the population score covariance C (even fold),
    via central FD on the smooth analytical density. Cached to disk."""
    import os
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    cache_file = os.path.join(
        cache_dir,
        f"even_fold_V_KL_g{GAMMA_EF:.2f}_top{r_max}_Nref{N_ref}_h{h_fd:.0e}_seed{seed}.npz",
    )
    if os.path.exists(cache_file):
        return np.load(cache_file)["V_KL"]

    rng = np.random.default_rng(seed)
    z = inv_logit_phi(sample_even_fold_noise(N_ref, rng))
    d = z.shape[1]
    T = np.empty_like(z)
    for j in range(d):
        zp = z.copy(); zp[:, j] += h_fd
        zm = z.copy(); zm[:, j] -= h_fd
        T[:, j] = (_log_copula_density_even_fold(zp) - _log_copula_density_even_fold(zm)) / (2.0 * h_fd)
    C = T.T @ T / N_ref
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(-eigvals)
    V_KL = eigvecs[:, order][:, :r_max]
    if os.path.isdir(cache_dir):
        np.savez(cache_file, V_KL=V_KL, eigvals_top=eigvals[order][:r_max],
                 trC=float(np.trace(C)), r_max=r_max, N_ref=N_ref, h_fd=h_fd, seed=seed)
    return V_KL


_V_KL_EVEN_FOLD = None


def oracle_subspace_even_fold(r):
    """Reference active subspace: the exact copula active subspace V_r^C for the
    even-fold law, via cas.deployment_oracle (top-r eigenspace of the exact
    copula-score covariance from cas.analytic_oracle). This is the §5.1 / SM4
    recovery target and Stage-1-isolated-KL floor, distinct from the (K=4, q=2)
    chosen-estimator spectrum. The logistic-marginal FD V_KL is retained as
    _compute_V_KL_even_fold for diagnostics only."""
    from .deployment_oracle import deployment_oracle_subspace
    return deployment_oracle_subspace("even_fold", r)
