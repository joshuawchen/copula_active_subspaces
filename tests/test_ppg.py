"""Tests for ``cas.ppg``.

PPG (polynomial-perturbed Gaussian) is a noise law with an axis-pair-aligned
rank-r* active subspace and quartic perturbations.
These tests verify:

  - sampling and density evaluation are consistent (E_z[log c(z)] >= 0)
  - the rank-Gauss z-space marginals are standard normal
  - V_KL is analytically span{b_0, b_1, b_2} at r=3 and span{e_0..e_5} at r=6
  - FD score covariance has the predicted cliff structure
  - stage1_oracle_kl returns zero at V_KL r=6 and a finite positive value
    elsewhere, mirroring the cubic-banana exact-cliff behaviour
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cas import (
    sample_ppg_noise,
    log_ppg_density,
    oracle_subspace_ppg,
    D_PPG,
    R_PPG_PRIMARY,
    LAMBDAS_PPG,
    stage1_oracle_kl,
)


def _log_c_ppg(z):
    """log copula density of PPG in rank-Gauss z-space."""
    sig = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    eta = np.log(sig) - np.log1p(-sig)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_ppg_density(eta) - log_f_mar.sum(axis=-1)


def _sample_z_ppg(N, rng):
    eta = sample_ppg_noise(N, rng)
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(sig)


def test_ppg_constants():
    """Sanity check on module constants."""
    assert D_PPG == 20
    assert R_PPG_PRIMARY == 3
    assert len(LAMBDAS_PPG) == R_PPG_PRIMARY
    assert all(l > 0 for l in LAMBDAS_PPG)


def test_ppg_sample_shape_and_marginals():
    """Sampled eta has the right shape; z-space marginals are ~N(0,1)."""
    rng = np.random.default_rng(0)
    eta = sample_ppg_noise(5000, rng)
    assert eta.shape == (5000, D_PPG)
    assert np.all(np.isfinite(eta))

    # z = Phi^{-1}(sigm(eta)) should have ~N(0,1) marginals on every coord
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    z = stats.norm.ppf(sig)
    z_means = z.mean(axis=0)
    z_stds = z.std(axis=0)
    # With N=5000 samples, std error of mean is ~1/sqrt(N) = 0.014
    assert np.abs(z_means).max() < 0.1, f"z means out of range: {z_means}"
    assert np.all(np.abs(z_stds - 1.0) < 0.05), f"z stds out of range: {z_stds}"


def test_ppg_log_density_finite():
    """log_ppg_density returns finite values on sampled eta."""
    rng = np.random.default_rng(1)
    eta = sample_ppg_noise(1000, rng)
    ld = log_ppg_density(eta)
    assert ld.shape == (1000,)
    assert np.all(np.isfinite(ld))


def test_ppg_copula_nonneg_KL():
    """E_z[log c(z)] >= 0 (Shannon: KL of joint vs product of marginals)."""
    rng = np.random.default_rng(2)
    z = _sample_z_ppg(5000, rng)
    log_c = _log_c_ppg(z)
    # E[log c] is the mutual-info-like KL, should be non-negative.
    # Margin allows for MC error at N=5000.
    assert log_c.mean() > -0.05, f"E[log c] = {log_c.mean():.4f} should be >= 0"


def test_oracle_subspace_ppg_orthonormal():
    """oracle_subspace_ppg returns orthonormal columns at r=1..6."""
    for r in range(1, 2 * R_PPG_PRIMARY + 1):
        V = oracle_subspace_ppg(r)
        assert V.shape == (D_PPG, r)
        Gram = V.T @ V
        assert np.abs(Gram - np.eye(r)).max() < 1e-10, (
            f"V_KL at r={r} not orthonormal: max err = {np.abs(Gram - np.eye(r)).max():.2e}"
        )


def test_oracle_subspace_ppg_rejects_r_gt_active():
    """Oracle subspace rejects r > 2 * N_PAIRS (the exact active rank)."""
    with pytest.raises(ValueError, match=r"exceeds active rank"):
        oracle_subspace_ppg(2 * R_PPG_PRIMARY + 1)


def test_ppg_FD_eigval_cliff():
    """FD score covariance has the predicted cliff: top-3 ~ 0.5, next-3 ~ 0.12,
    bulk ~ 0; cliff l_3/l_4 ~ 4."""
    rng = np.random.default_rng(42)
    z = _sample_z_ppg(10000, rng)
    h = 1e-3
    T = np.empty_like(z)
    for j in range(D_PPG):
        zp = z.copy()
        zm = z.copy()
        zp[:, j] += h
        zm[:, j] -= h
        T[:, j] = (_log_c_ppg(zp) - _log_c_ppg(zm)) / (2 * h)
    C = T.T @ T / 10000
    eigs = np.sort(np.linalg.eigvalsh(C))[::-1]

    # Top-3 in a tight band around 0.5
    assert np.all((eigs[:3] > 0.4) & (eigs[:3] < 0.6)), (
        f"Top-3 eigvals out of band: {eigs[:3]}"
    )
    # Next-3 in a tight band around 0.12
    assert np.all((eigs[3:6] > 0.08) & (eigs[3:6] < 0.16)), (
        f"Eigvals 4-6 out of band: {eigs[3:6]}"
    )
    # Eigvals 7+ near zero
    assert eigs[6] < 1e-6, f"Bulk eigval l_7 = {eigs[6]:.2e} should be ~0"
    # Cliff l_3/l_4 ~ 4
    ratio = eigs[2] / eigs[3]
    assert 2.5 < ratio < 6.0, f"Cliff l_3/l_4 = {ratio:.2f} out of expected range"


def test_ppg_FD_eigvecs_match_analytical():
    """Top-3 and top-6 FD eigvectors agree with the analytical V_KL."""
    rng = np.random.default_rng(42)
    z = _sample_z_ppg(10000, rng)
    h = 1e-3
    T = np.empty_like(z)
    for j in range(D_PPG):
        zp = z.copy()
        zm = z.copy()
        zp[:, j] += h
        zm[:, j] -= h
        T[:, j] = (_log_c_ppg(zp) - _log_c_ppg(zm)) / (2 * h)
    C = T.T @ T / 10000
    w, V_fd = np.linalg.eigh(C)
    order = np.argsort(-w)
    V_fd_3 = V_fd[:, order][:, :3]
    V_fd_6 = V_fd[:, order][:, :6]

    V_an_3 = oracle_subspace_ppg(3)
    V_an_6 = oracle_subspace_ppg(6)

    def sin_theta(A, B):
        sv = np.linalg.svd(A.T @ B)[1]
        return np.sqrt(max(0, A.shape[1] - (sv ** 2).sum()))

    # At r=3: analytical and FD agree to ~MC noise
    assert sin_theta(V_fd_3, V_an_3) < 0.1, (
        f"sin Theta(FD V_KL r=3, analytical) = {sin_theta(V_fd_3, V_an_3):.3f}"
    )
    # At r=6: exact agreement (full active subspace)
    assert sin_theta(V_fd_6, V_an_6) < 0.01, (
        f"sin Theta(FD V_KL r=6, analytical) = {sin_theta(V_fd_6, V_an_6):.4f}"
    )


def test_ppg_VKL_r6_gives_zero_KL():
    """V_KL at r=6 captures the full active subspace: stage1 KL = 0 exactly."""
    V_an_6 = oracle_subspace_ppg(6)
    out = stage1_oracle_kl(
        V_an_6, _log_c_ppg, _sample_z_ppg,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(0),
    )
    assert abs(out["kl_est"]) < 1e-6, (
        f"KL at exact V_KL r=6 should be ~0, got {out['kl_est']:.4e}"
    )
    # ESS = M_inner means perfect equal weights (the c_r(u) is exact)
    assert out["ess_min"] == pytest.approx(out["M_inner"], rel=1e-9)


def test_ppg_VKL_r3_below_random():
    """V_KL at r=3 gives KL strictly less than random V_r at r=3."""
    V_an_3 = oracle_subspace_ppg(3)
    out_kl = stage1_oracle_kl(
        V_an_3, _log_c_ppg, _sample_z_ppg,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(0),
    )
    rng = np.random.default_rng(99)
    A = rng.standard_normal((D_PPG, 3))
    V_rand, _ = np.linalg.qr(A)
    out_rand = stage1_oracle_kl(
        V_rand, _log_c_ppg, _sample_z_ppg,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(1),
    )
    # Should be a meaningful gap (V_KL captures 80% of trace, random doesn't)
    assert out_kl["kl_est"] < out_rand["kl_est"] - 0.05, (
        f"V_KL KL={out_kl['kl_est']:.3f} should be << random KL={out_rand['kl_est']:.3f}"
    )
    # And both should be non-negative
    assert out_kl["kl_est"] > -0.05
    assert out_rand["kl_est"] > -0.05
