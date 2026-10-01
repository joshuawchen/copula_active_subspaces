"""Tests for ``cas.oracle_reduced``.

Sanity tests of the Stage-1-isolated KL estimator on cubic-banana (where
the exact value is known by construction) and banana (where the value is
bounded by the Cor 4.4 bound).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cas import sample_noise, log_noise_density, oracle_subspace
from cas.cubic_banana import (
    sample_cubic_banana_noise,
    log_cubic_banana_density,
    oracle_subspace_cubic_banana,
    D_CB,
    R_CB_TARGET,
)
from cas.oracle_reduced import stage1_oracle_kl


def _log_c_banana(z):
    sig = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    eta = np.log(sig) - np.log1p(-sig)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_noise_density(eta) - log_f_mar.sum(axis=-1)


def _sample_z_banana(N, rng):
    eta = sample_noise(N, rng)
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(sig)


def _log_c_cb(z):
    sig = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    eta = np.log(sig) - np.log1p(-sig)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_cubic_banana_density(eta) - log_f_mar.sum(axis=-1)


def _sample_z_cb(N, rng):
    eta = sample_cubic_banana_noise(N, rng)
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(sig)


def test_cubic_banana_VKL_exact_zero():
    """Cubic-banana V_KL at r=6 is the exact rank — KL should be machine zero."""
    V_KL = oracle_subspace_cubic_banana(R_CB_TARGET)
    out = stage1_oracle_kl(
        V_KL, _log_c_cb, _sample_z_cb,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(0),
    )
    # KL is exactly zero in expectation; the MC estimator should be O(1e-10)
    assert abs(out["kl_est"]) < 1e-6, (
        f"KL at exact V_KL on cubic-banana should be ~0, got {out['kl_est']:.4e}"
    )
    # When c depends only on V_r^T z, ESS = M_inner (perfect equal weights)
    assert out["ess_min"] == pytest.approx(out["M_inner"], rel=1e-9), (
        f"ESS at exact V_KL should equal M_inner; got "
        f"ESS_min={out['ess_min']}, M_inner={out['M_inner']}"
    )


def test_banana_VKL_below_cor44_bound():
    """Banana V_KL: KL should be below the Cor 4.4 bound of 0.65 nats."""
    V_KL = oracle_subspace(4)
    out = stage1_oracle_kl(
        V_KL, _log_c_banana, _sample_z_banana,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(0),
    )
    # Cor 4.4 bound at V_KL on banana ~ 0.65. Actual KL should be strictly less.
    # MC estimator with 200 outer samples has SE of order 0.05-0.10, so check
    # within a reasonable margin.
    assert out["kl_est"] < 1.0, (
        f"KL at V_KL on banana should be below Cor 4.4 bound + margin, got {out['kl_est']:.3f}"
    )
    assert out["kl_est"] > 0.0, (
        f"KL must be non-negative, got {out['kl_est']:.3f}"
    )


def test_random_VR_higher_than_VKL():
    """Random V_r should give higher KL than V_KL on cubic-banana."""
    V_KL = oracle_subspace_cubic_banana(R_CB_TARGET)
    rng_rand = np.random.default_rng(11)
    A = rng_rand.standard_normal((D_CB, R_CB_TARGET))
    V_rand, _ = np.linalg.qr(A)

    out_KL = stage1_oracle_kl(
        V_KL, _log_c_cb, _sample_z_cb,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(0),
    )
    out_rand = stage1_oracle_kl(
        V_rand, _log_c_cb, _sample_z_cb,
        N_outer=200, M_inner=512,
        rng=np.random.default_rng(1),
    )
    # V_rand should be strictly worse (higher KL) than V_KL.
    # Use a generous margin since both have MC noise.
    assert out_rand["kl_est"] > out_KL["kl_est"] + 0.1, (
        f"Random V_r should give higher KL: V_KL={out_KL['kl_est']:.3f}, "
        f"V_rand={out_rand['kl_est']:.3f}"
    )


def test_non_orthonormal_VR_rejected():
    """V_r with non-orthonormal columns should be rejected."""
    V_bad = np.random.default_rng(0).standard_normal((20, 4))  # not orthonormal
    with pytest.raises(ValueError, match="orthonormal"):
        stage1_oracle_kl(
            V_bad, _log_c_banana, _sample_z_banana,
            N_outer=10, M_inner=64,
            rng=np.random.default_rng(0),
        )
