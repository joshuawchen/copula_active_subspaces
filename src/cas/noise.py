"""Example 1 of Part I. The outputs are the saturated real and imaginary parts
of (W0 + i W1)^2,

    W2 = c tanh((W0^2 - W1^2) / M) + sqrt(1 - a^2) eps2,
    W3 = c tanh(2 W0 W1 / M) + sqrt(1 - a^2) eps3,

with M = 2.5 and a = 0.7, mixed by the rotation of cas.mixing and mapped by
eta = logit(Phi(z)) (cas.transforms).
"""
from __future__ import annotations
import numpy as np
from scipy import stats

from .config import D_OBS, K_BAN, A_BAN, RHO
from .mixing import block_rotation, B_SUPPORT
from .transforms import logit_phi, inv_logit_phi, logit_phi_log_jacobian_z


# Latent mixing: the orthogonal rotation of cas.mixing on the first B_SUPPORT
# coordinates.
# log|det Q| = 0, so _LD_LOG_DET_L below is 0 by construction.
L_CHOL = block_rotation(D_OBS, B_SUPPORT)


# ---------------------------------------------------------------------------
# Conformal z^2 block constants (single source of truth for the active law).
# ---------------------------------------------------------------------------
# Conditional variance s2 = 1 - a^2 and the saturation scale M. The amplitudes
# c2, c3 are variance-normalized so Var(mu2) = Var(mu3) = a^2; because
# (Re z^2, Im z^2) = r^2 (cos 2theta, sin 2theta) with theta uniform, the two
# arguments are identically distributed, so Var(g2) = Var(g3) and c2 = c3.
_CONF_M: float = 2.5            # saturation scale (tanh knee of the z^2 argument)
_CONF_ALPHA: float = A_BAN      # conditional-mean variance fraction (Var(mu) = a^2)


def _conf_g2(W0, W1):
    """Re(z^2)/M argument fed through tanh: tanh((W0^2 - W1^2)/M)."""
    return np.tanh((W0 * W0 - W1 * W1) / _CONF_M)


def _conf_g3(W0, W1):
    """Im(z^2)/M argument fed through tanh: tanh(2 W0 W1 / M)."""
    return np.tanh(2.0 * W0 * W1 / _CONF_M)


def _conf_amplitude(gf, n: int = 129) -> float:
    """c = a / sqrt(Var_{W0,W1~N(0,1)}[gf(W0,W1)]) via 2-D Gauss-Hermite.

    gf is odd under the relevant reflection, so E[gf] = 0 and Var = E[gf^2];
    the amplitude makes Var(c gf) = a^2 (so the output has unit variance).
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    a = x[:, None]; b = x[None, :]
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    wt = (w[:, None] * w[None, :]) * (norm ** 2)
    v = gf(a, b)
    mean = float((wt * v).sum())
    var = float((wt * v * v).sum()) - mean * mean
    return float(_CONF_ALPHA / np.sqrt(var))


_CONF_C2: float = _conf_amplitude(_conf_g2)   # = 1.476 at M=2.5, a=0.7
_CONF_C3: float = _conf_amplitude(_conf_g3)   # = _CONF_C2 by symmetry


def _conf_means(W0, W1):
    """Conditional means (mu2, mu3) = (c2 tanh(Re z^2 / M), c3 tanh(Im z^2 / M))."""
    return _CONF_C2 * _conf_g2(W0, W1), _CONF_C3 * _conf_g3(W0, W1)


def sample_banana_W(N, rng):
    """Latent W: conformal z^2 block at coords 0..3, then i.i.d. N(0,1) padding.

    coords (0, 1) are the inputs (W0, W1); coords (2, 3) the conformal outputs
    (W2, W3). Sampler and density share one bend by construction (_conf_means).
    """
    W = np.empty((N, D_OBS))
    s = np.sqrt(1.0 - A_BAN ** 2)
    W0 = rng.standard_normal(N)
    W1 = rng.standard_normal(N)
    mu2, mu3 = _conf_means(W0, W1)
    W[:, 0] = W0
    W[:, 1] = W1
    W[:, 2] = mu2 + s * rng.standard_normal(N)
    W[:, 3] = mu3 + s * rng.standard_normal(N)
    W[:, 4:] = rng.standard_normal((N, D_OBS - 4))
    return W


def _logit_cdf(Zlat):
    """Numerically stable eta = logit(Phi(Zlat)) without clipping."""
    return logit_phi(Zlat)


def _invlogit_to_Z(eta):
    """Numerically stable Z = Phi^{-1}(sigmoid(eta)) without clipping."""
    return inv_logit_phi(eta)


def sample_noise(N, rng):
    """Draw N i.i.d. eta from the conformal-z^2 banana noise."""
    W = sample_banana_W(N, rng)
    Zlat = W @ L_CHOL.T
    return _logit_cdf(Zlat)


# Precomputed constants for log_noise_density (hot path; called inside the
# nested-MC inner loop in cas.oracle_reduced, ~1500 times per stage1 call).
_LD_LOG_2PI = float(np.log(2.0 * np.pi))
_LD_S2 = 1.0 - A_BAN ** 2
_LD_LOG_S2 = float(np.log(_LD_S2))
_LD_INV_S2 = 1.0 / _LD_S2
_LD_LOG_DET_L = float(np.linalg.slogdet(L_CHOL)[1])   # = 0 for an orthogonal Q


# ---------------------------------------------------------------------------
# Single source of truth for the active conformal-z^2 block (W-space).
# ---------------------------------------------------------------------------
# The active coordinates are the first 2*K_BAN = 4 entries of W: inputs
# (W0, W1) at columns 0,1 and outputs (W2, W3) at columns 2,3, with
#     W0, W1 ~ N(0, 1),   W2 | . ~ N(mu2, s2),   W3 | . ~ N(mu3, s2),
#     mu2 = c2 tanh((W0^2 - W1^2)/M),  mu3 = c3 tanh(2 W0 W1 / M).
# Both log_noise_density (below) and cas.oracle_kl_marginalization consume
# these functions, so the deployed law lives in exactly one place and the
# marginalizer cannot drift from the sampler. To change the block edit only
# _conf_g2 / _conf_g3 (and re-derive the grad/Hessian pieces).
_BAN_S2 = _LD_S2            # conditional variance of W2, W3 (= 1 - a^2)
_BAN_LOG_S2 = _LD_LOG_S2    # = log(_BAN_S2), precomputed


def log_pi_W_active(W_act):
    """log pi for the conformal z^2 active block (no Gaussian residual, no
    Jacobian, no Toeplitz mixing). W_act: (N, 4) = [W0, W1, W2, W3]. Single
    source of truth shared with the marginalizer.

    The four-coordinate normalization (-2 log 2pi - log s2) matches the old
    two-pair total because both have two unit-variance and two s2-variance
    Gaussian factors.
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    mu2, mu3 = _conf_means(W0, W1)
    return (
        -0.5 * (W0 * W0 + W1 * W1)
        - 0.5 * (W2 - mu2) ** 2 / _BAN_S2
        - 0.5 * (W3 - mu3) ** 2 / _BAN_S2
        - _BAN_LOG_S2
        - 2.0 * _LD_LOG_2PI
    )


def grad_log_pi_W_active(W_act):
    """grad log pi_W_active wrt W_act, returns (N, 4).

        d/dW0 = -W0 + r2 dmu2/dW0 + r3 dmu3/dW0
        d/dW1 = -W1 + r2 dmu2/dW1 + r3 dmu3/dW1
        d/dW2 = -r2,   d/dW3 = -r3,   r_k = (W_k - mu_k) / s2
    with dmu2/dW0 = c2 sech^2(u2) (2 W0/M), dmu2/dW1 = c2 sech^2(u2)(-2 W1/M),
         dmu3/dW0 = c3 sech^2(u3) (2 W1/M), dmu3/dW1 = c3 sech^2(u3)(2 W0/M),
         u2 = (W0^2 - W1^2)/M, u3 = 2 W0 W1 / M.
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    t2 = _conf_g2(W0, W1); t3 = _conf_g3(W0, W1)
    mu2 = _CONF_C2 * t2; mu3 = _CONF_C3 * t3
    sh2 = 1.0 - t2 * t2; sh3 = 1.0 - t3 * t3      # sech^2(u)
    d2_0 = _CONF_C2 * sh2 * (2.0 * W0 / _CONF_M)
    d2_1 = _CONF_C2 * sh2 * (-2.0 * W1 / _CONF_M)
    d3_0 = _CONF_C3 * sh3 * (2.0 * W1 / _CONF_M)
    d3_1 = _CONF_C3 * sh3 * (2.0 * W0 / _CONF_M)
    r2 = (W2 - mu2) / _BAN_S2
    r3 = (W3 - mu3) / _BAN_S2
    grad = np.empty_like(W_act)
    grad[:, 0] = -W0 + r2 * d2_0 + r3 * d3_0
    grad[:, 1] = -W1 + r2 * d2_1 + r3 * d3_1
    grad[:, 2] = -r2
    grad[:, 3] = -r3
    return grad


def hess_log_pi_W_active(W_act):
    """Hessian of log pi_W_active wrt W_act, returns (N, 4, 4).

    Input-input block (0,1) carries the conformal curvature; the input-output
    cross terms are dmu_k/dW_i / s2; the output-output block is -I/s2.
    Second derivatives of the means (a.. for mu2, b.. for mu3):
        a00 = c2 sech^2 (-2 t2 (2 W0/M)^2 + 2/M)
        a11 = c2 sech^2 (-2 t2 (2 W1/M)^2 - 2/M)
        a01 = c2 sech^2 t2 (8 W0 W1 / M^2)
        b00 = -2 c3 sech^2 t3 (2 W1/M)^2
        b11 = -2 c3 sech^2 t3 (2 W0/M)^2
        b01 = c3 sech^2 (-2 t3 (4 W0 W1/M^2) + 2/M)
    """
    W_act = np.atleast_2d(np.asarray(W_act, dtype=float))
    N = W_act.shape[0]
    W0 = W_act[:, 0]; W1 = W_act[:, 1]; W2 = W_act[:, 2]; W3 = W_act[:, 3]
    t2 = _conf_g2(W0, W1); t3 = _conf_g3(W0, W1)
    mu2 = _CONF_C2 * t2; mu3 = _CONF_C3 * t3
    sh2 = 1.0 - t2 * t2; sh3 = 1.0 - t3 * t3
    d2_0 = _CONF_C2 * sh2 * (2.0 * W0 / _CONF_M)
    d2_1 = _CONF_C2 * sh2 * (-2.0 * W1 / _CONF_M)
    d3_0 = _CONF_C3 * sh3 * (2.0 * W1 / _CONF_M)
    d3_1 = _CONF_C3 * sh3 * (2.0 * W0 / _CONF_M)
    a00 = _CONF_C2 * sh2 * (-2.0 * t2 * (2.0 * W0 / _CONF_M) ** 2 + 2.0 / _CONF_M)
    a11 = _CONF_C2 * sh2 * (-2.0 * t2 * (2.0 * W1 / _CONF_M) ** 2 - 2.0 / _CONF_M)
    a01 = _CONF_C2 * sh2 * t2 * (8.0 * W0 * W1 / _CONF_M ** 2)
    b00 = -2.0 * _CONF_C3 * sh3 * t3 * (2.0 * W1 / _CONF_M) ** 2
    b11 = -2.0 * _CONF_C3 * sh3 * t3 * (2.0 * W0 / _CONF_M) ** 2
    b01 = _CONF_C3 * sh3 * (-2.0 * t3 * (4.0 * W0 * W1 / _CONF_M ** 2) + 2.0 / _CONF_M)
    r2 = (W2 - mu2) / _BAN_S2
    r3 = (W3 - mu3) / _BAN_S2
    H = np.zeros((N, 4, 4))
    H[:, 0, 0] = -1.0 - d2_0 ** 2 / _BAN_S2 + r2 * a00 - d3_0 ** 2 / _BAN_S2 + r3 * b00
    H[:, 1, 1] = -1.0 - d2_1 ** 2 / _BAN_S2 + r2 * a11 - d3_1 ** 2 / _BAN_S2 + r3 * b11
    H[:, 0, 1] = H[:, 1, 0] = (
        -d2_1 * d2_0 / _BAN_S2 + r2 * a01 - d3_1 * d3_0 / _BAN_S2 + r3 * b01
    )
    H[:, 2, 2] = -1.0 / _BAN_S2
    H[:, 3, 3] = -1.0 / _BAN_S2
    H[:, 0, 2] = H[:, 2, 0] = d2_0 / _BAN_S2
    H[:, 1, 2] = H[:, 2, 1] = d2_1 / _BAN_S2
    H[:, 0, 3] = H[:, 3, 0] = d3_0 / _BAN_S2
    H[:, 1, 3] = H[:, 3, 1] = d3_1 / _BAN_S2
    return H


def log_noise_density(eta):
    """Closed-form log pi_eta(eta) for the conformal-z^2 banana law.

    eta -> Zlat via inverse logit (stable), Zlat -> W via L^{-1} Zlat,
    log pi_eta = log pi_W(W) - log|det L| + sum_i log|d Zlat_i / d eta_i|.

    Stable for |eta| up to ~700 (no clipping).
    """
    eta = np.asarray(eta, dtype=float)
    flat = (eta.ndim == 1)
    if flat:
        eta = eta[None, :]

    # Stable inverse: eta -> Zlat without clipping.
    Zlat = inv_logit_phi(eta)
    W = np.linalg.solve(L_CHOL, Zlat.T).T

    # log pi_W active block: delegate to the single-source-of-truth active
    # density (cas.oracle_kl_marginalization imports the same log_pi_W_active).
    log_pi_block = log_pi_W_active(W[:, :2 * K_BAN])
    W_rest = W[:, 2 * K_BAN:]
    n_rest = D_OBS - 2 * K_BAN
    log_pi_rest = (
        -0.5 * (W_rest * W_rest).sum(axis=1)
        - 0.5 * n_rest * _LD_LOG_2PI
    )
    log_pi_W = log_pi_block + log_pi_rest

    # log Jacobian of the componentwise logit-Phi transform per coord:
    #   log|d Zlat_j / d eta_j| = log_ndtr(Zlat_j) + log_ndtr(-Zlat_j)
    #                             + 0.5 Zlat_j^2 + 0.5 log(2 pi)
    # computed stably by logit_phi_log_jacobian_z(Zlat).
    log_jac_logistic = logit_phi_log_jacobian_z(Zlat).sum(axis=1)

    out = log_pi_W - _LD_LOG_DET_L + log_jac_logistic
    return out[0] if flat else out


def oracle_subspace_geom(r):
    """Geometric active subspace: orthonormalized span of first r columns of L.

    span(V_geom) = L * span(e_1, ..., e_r): the latent non-Gaussian subspace
    transported through the mixing L. This is NOT the top-r eigenspace of the
    rank-Gaussianized copula's score covariance C (= V_KL, returned by
    oracle_subspace). sinTheta(V_geom, V_KL) for the conformal law is
    regenerated by the verification run.

    Retained for diagnostics and provenance; do not use as the oracle
    reference in KL-themed experiments.
    """
    V = L_CHOL[:, :r].copy()
    Q, _ = np.linalg.qr(V)
    return Q


# ---------------------------------------------------------------------------
# V_KL = top-r eigvecs of the population score covariance C
# ---------------------------------------------------------------------------
#
# Computed once via numerical-FD on the smooth log_noise_density, cached to
# disk keyed on (N_ref, h, seed). The conformal-z^2 density is C^inf analytic
# (no MC tables), so central finite differences converge cleanly at h ~ 1e-4.
# The resulting V_KL is the rank-r minimizer of the Cor 4.4 bound
# 0.5 * tr((I - V V^T) C).

_VKL_N_REF: int = 30_000
_VKL_H_FD: float = 1e-4
_VKL_SEED: int = 20260513
_VKL_R_MAX: int = 8  # cache top-8 eigenvectors; oracle_subspace(r) slices


def _log_copula_density_banana(z):
    """log c(z) = log pi_eta(eta(z)) - sum_j log f^mar_j(eta_j(z)), where
    eta_j = logit(Phi(z_j)) and f^mar_j is taken as the standard-logistic
    density.

    Inherited approximation: f^mar_j = standard logistic. The conformal law's
    active marginals are symmetric and light-tailed (bounded mean plus
    Gaussian), so this is a mild approximation; the verification run's
    Stein-identity check confirms it. The exact §5.1 subspace is the analytic
    copula-score covariance (cas.analytic_oracle), not this FD estimator.

    Computed via stable logit_phi (no clipping). For |z| > ~37 the logit_phi
    result is +/-inf; downstream consumers should filter.
    """
    eta = logit_phi(z)
    # Standard-logistic density: log f(eta) = -eta - 2 log(1 + e^{-eta})
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_noise_density(eta) - log_f_mar.sum(axis=-1)


def _compute_V_KL_banana(
    r_max: int = _VKL_R_MAX,
    N_ref: int = _VKL_N_REF,
    h_fd: float = _VKL_H_FD,
    seed: int = _VKL_SEED,
) -> np.ndarray:
    """Estimate top-r_max eigvecs of population score covariance C (conformal).

    Returns V_KL of shape (d, r_max), columns sorted by eigenvalue descending.

    Cached to disk; rebuild forced by changing any of (r_max, N_ref, h_fd, seed)
    or the cache filename (keyed '_confz2' for this law).
    """
    import os
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
    cache_dir = os.path.abspath(cache_dir)
    cache_file = os.path.join(
        cache_dir,
        f"banana_V_KL_confz2_top{r_max}_Nref{N_ref}_h{h_fd:.0e}_seed{seed}.npz",
    )
    if os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["V_KL"]

    rng = np.random.default_rng(seed)
    eta = sample_noise(N_ref, rng)
    # Stable inverse: eta -> z without clipping.
    z = inv_logit_phi(eta)
    d = z.shape[1]

    # Numerical-FD score (central differences).
    T = np.empty_like(z)
    for j in range(d):
        zp = z.copy(); zp[:, j] += h_fd
        zm = z.copy(); zm[:, j] -= h_fd
        T[:, j] = (
            _log_copula_density_banana(zp)
            - _log_copula_density_banana(zm)
        ) / (2.0 * h_fd)

    C = T.T @ T / N_ref
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(-eigvals)
    V_KL = eigvecs[:, order][:, :r_max]
    eigvals_top = eigvals[order][:r_max]
    trC = float(np.trace(C))

    if os.path.isdir(cache_dir):
        np.savez(
            cache_file,
            V_KL=V_KL,
            eigvals_top=eigvals_top,
            trC=trC,
            r_max=r_max,
            N_ref=N_ref,
            h_fd=h_fd,
            seed=seed,
        )
    return V_KL


# Cache V_KL at module load (or first call) at the deployment r_max.
_V_KL_BANANA = None


def oracle_subspace(r):
    """Reference active subspace: the exact copula active subspace V_r^C.

    Delegates to cas.deployment_oracle.deployment_oracle_subspace, which
    returns the top-r eigenspace of the exact copula-score covariance C
    (cas.analytic_oracle: exact Gaussian-mixture marginals of the mixed
    coordinates -> exact Z-space score -> C = mean_i S S^T), the reference subspace of
    Part I, Section 5.

    The logistic-marginal finite-difference V_KL (_compute_V_KL_banana,
    N_ref=30,000) is a different, approximate object, retained for diagnostics
    only (cf. oracle_subspace_geom for the orth(L[:, :r]) definition).
    """
    from .deployment_oracle import deployment_oracle_subspace
    return deployment_oracle_subspace("banana", r)
