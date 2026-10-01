"""Logistic-base marginal density estimator for the nested Monte Carlo
reference computations.

Each coordinate is mapped to w = Phi^{-1}(F_logistic(eta)) and fitted by
Hyvarinen score matching in the family p(w) proportional to
phi(w) exp(sum_k theta_k He_k(w)), k in {2, 4, 6}; a secondary transform
gives standard-normal marginals.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_HE_DEGREES = (2, 4)           # even symmetric tilt (variance + kurtosis).
# He6 is deliberately excluded: a degree-6 polynomial in the exponent produces
# tail spikes that collapse the nested-MC inner ESS (the KDE failure mode). The
# logistic deviation is a symmetric kurtosis shift, captured at degree 4; higher
# even terms add only noise-level moments at the cost of stability.
_FIT_N = 2_000_000             # samples for the per-coord score-match fit
_FIT_SEED = 20260601
_GRID_LO, _GRID_HI, _GRID_N = -12.0, 12.0, 8001   # w-grid for normalizer + CDF
_ACTIVE_TOL = 3e-3             # |theta| below this (~ 4x the 2M-sample noise
                               # floor) -> coord treated as exact logistic
_WIN_W0, _WIN_W1 = 3.5, 6.0    # tilt applied in full for |w|<w0, raised-cosine
                               # tapered to 0 by |w|>w1, so the density reverts
                               # to the normal base in the (data-sparse) tails.
                               # This keeps log(phi/p) bounded -> the nested-MC
                               # inner sampler stays conditioned even when the
                               # marginal is leptokurtic (theta_4 > 0), where an
                               # un-windowed polynomial tilt is non-integrable.
_CLIP = 1e-12


def _he_norm(x, k):
    """Orthonormal probabilists' Hermite He_k(x) / sqrt(k!) via recurrence."""
    if k == 0:
        return np.ones_like(x)
    He = [np.ones_like(x), x.copy()]
    for n in range(1, k):
        He.append(x * He[n] - n * He[n - 1])
    return He[k] / math.sqrt(float(math.factorial(k)))


def _sigmoid(e):
    return 1.0 / (1.0 + np.exp(-e))


def _eta_to_w(eta):
    """w = Phi^{-1}(F_logistic(eta)) = Phi^{-1}(sigmoid(eta)), per coordinate."""
    return norm.ppf(np.clip(_sigmoid(eta), _CLIP, 1.0 - _CLIP))


def _window(w):
    """Raised-cosine taper: 1 for |w|<w0, 0 for |w|>w1, smooth between."""
    a = np.abs(np.asarray(w, dtype=float))
    out = np.ones_like(a)
    mid = (a > _WIN_W0) & (a < _WIN_W1)
    out[mid] = 0.5 * (1.0 + np.cos(np.pi * (a[mid] - _WIN_W0)
                                  / (_WIN_W1 - _WIN_W0)))
    out[a >= _WIN_W1] = 0.0
    return out


def _tilt(theta_j, w):
    """Windowed Hermite tilt g(w) = [sum_k theta_k He_k^norm(w)] * window(w)."""
    g = np.zeros_like(np.asarray(w, dtype=float))
    for i, k in enumerate(_HE_DEGREES):
        g = g + theta_j[i] * _he_norm(w, k)
    return g * _window(w)


def _fit_theta_scorematch(w):
    """Hyvarinen score-match of the even Hermite tilt on a 1-D sample w.

    Model log p(w) = -w^2/2 + sum_k theta_k psi_k(w) - A, psi_k = He_k^norm,
    score s(w) = -w + sum_k theta_k psi_k'(w) with psi_k' = sqrt(k) He_{k-1}^norm
    and psi_k'' = sqrt(k(k-1)) He_{k-2}^norm. The minimizer solves M theta = c,
    M_kl = E[psi_k' psi_l'], c_k = E[w psi_k' - psi_k'']. By Stein's identity
    c_k = 0 when w ~ N(0,1), so theta = 0 recovers the logistic marginal.
    """
    K = len(_HE_DEGREES)
    n = w.shape[0]
    dpsi = np.empty((n, K))
    d2psi = np.empty((n, K))
    for i, k in enumerate(_HE_DEGREES):
        dpsi[:, i] = math.sqrt(k) * _he_norm(w, k - 1)
        d2psi[:, i] = math.sqrt(k * (k - 1)) * _he_norm(w, k - 2)
    M = dpsi.T @ dpsi / n
    c = (w[:, None] * dpsi - d2psi).mean(axis=0)
    return np.linalg.solve(M + 1e-10 * np.eye(K), c)


@dataclass
class ConsistentMarginalFit:
    """Per-coordinate logistic-base Hermite-tilt marginal fit.

    theta:  (d, K) score-matched even-Hermite tilt coefficients (0 on inactive).
    Acoef:  (d,)   log-normalizers A_j = log E_phi[exp(g_j)] (0 on inactive).
    grid:   (G,)    shared w-grid for the tabulated corrected CDF P_j.
    cdf:    (d, G)  P_j(grid) per coordinate (Phi on inactive coords).
    active: (d,)    bool, max_k|theta_jk| > _ACTIVE_TOL.
    """
    theta: np.ndarray
    Acoef: np.ndarray
    grid: np.ndarray
    cdf: np.ndarray
    active: np.ndarray

    def _g(self, w, j):
        return _tilt(self.theta[j], w)

    def w_to_z(self, w):
        """Secondary rank-Gauss z_j = Phi^{-1}(P_j(w_j)); identity if inactive."""
        z = w.copy()
        for j in np.nonzero(self.active)[0]:
            u = np.interp(w[:, j], self.grid, self.cdf[j])
            z[:, j] = norm.ppf(np.clip(u, _CLIP, 1.0 - _CLIP))
        return z

    def z_to_w(self, z):
        """Inverse w_j = P_j^{-1}(Phi(z_j)); identity if inactive."""
        w = z.copy()
        if not self.active.any():
            return w
        u = np.clip(norm.cdf(z), _CLIP, 1.0 - _CLIP)
        for j in np.nonzero(self.active)[0]:
            w[:, j] = np.interp(u[:, j], self.cdf[j], self.grid)
        return w

    def correction(self, w):
        """sum_j [ g_j(w_j) - A_j ] over active coords (0 elsewhere)."""
        corr = np.zeros(w.shape[0])
        for j in np.nonzero(self.active)[0]:
            corr = corr + self._g(w[:, j], j) - self.Acoef[j]
        return corr


def _build_cdf(theta_j, grid):
    g = _tilt(theta_j, grid)
    dens = norm.pdf(grid) * np.exp(g)
    Z = float(np.trapezoid(dens, grid))
    dens = dens / Z
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf = cdf / cdf[-1]
    return math.log(Z), cdf


def fit_consistent_marginals(sample_eta, cache_key, n_fit=_FIT_N,
                             seed=_FIT_SEED, cache_dir=None):
    """Fit (and disk-cache) the per-coordinate logistic-base Hermite tilt.

    sample_eta(N, rng) -> (N, d) draws from the benchmark noise. cache_key is a
    short benchmark tag; the fit is keyed on (cache_key, n_fit, degrees, seed).
    """
    if cache_dir is None:
        cache_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    deg = "".join(str(k) for k in _HE_DEGREES)
    path = os.path.join(
        cache_dir,
        f"consistent_marginal_{cache_key}_He{deg}_"
        f"w{_WIN_W0:.1f}-{_WIN_W1:.1f}_N{n_fit}_seed{seed}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return ConsistentMarginalFit(d["theta"], d["Acoef"], d["grid"],
                                     d["cdf"], d["active"])

    eta = sample_eta(n_fit, np.random.default_rng(seed))
    dobs = eta.shape[1]
    grid = np.linspace(_GRID_LO, _GRID_HI, _GRID_N)
    theta = np.zeros((dobs, len(_HE_DEGREES)))
    Acoef = np.zeros(dobs)
    cdf = np.tile(norm.cdf(grid), (dobs, 1))
    active = np.zeros(dobs, dtype=bool)
    for j in range(dobs):
        w = _eta_to_w(eta[:, j])
        th = _fit_theta_scorematch(w)
        if np.max(np.abs(th)) > _ACTIVE_TOL:
            active[j] = True
            theta[j] = th
            Acoef[j], cdf[j] = _build_cdf(th, grid)
    if os.path.isdir(cache_dir):
        np.savez(path, theta=theta, Acoef=Acoef, grid=grid, cdf=cdf, active=active)
    return ConsistentMarginalFit(theta, Acoef, grid, cdf, active)


def make_log_c_z(fit, log_eta_density):
    """Closed-form log copula density log c(z) for the consistent estimator.

    log c(z) = log c_logistic(w(z)) - sum_j [ g_j(w_j) - A_j ], w_j = P_j^{-1}
    (Phi(z_j)); c_logistic is the logistic-marginal copula evaluated at w.
    """
    def log_c_z(z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        w = fit.z_to_w(z)
        sig = np.clip(norm.cdf(w), _CLIP, 1.0 - _CLIP)
        eta = np.log(sig) - np.log1p(-sig)
        log_f = -eta - 2.0 * np.log1p(np.exp(-eta))
        return log_eta_density(eta) - log_f.sum(axis=-1) - fit.correction(w)
    return log_c_z


def make_sample_z(fit, sample_eta):
    """Draw z ~ pi_Z in the consistent (exactly-normal-marginal) frame."""
    def sample_z(N, rng):
        w = _eta_to_w(sample_eta(N, rng))
        return fit.w_to_z(w)
    return sample_z
