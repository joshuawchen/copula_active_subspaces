"""Tests for ``cas.stage1`` (Algorithm 1).

Verifies that ``cas_fit`` and ``cas_fit_streaming`` agree, that the fitted
V_r is orthonormal, and that on the d=20 banana noise law the recovered
subspace is close to the analytic oracle at large N.
"""
from __future__ import annotations

import numpy as np
import pytest

from cas import (
    cas_fit,
    cas_fit_streaming,
    sample_noise,
    sin_theta_F,
    oracle_subspace,
)
from cas.config import R_ORACLE


def test_cas_fit_streaming_agrees_with_cas_fit():
    """At fixed seed, the two entry points should produce equivalent V_r."""
    rng = np.random.default_rng(0)
    eta = sample_noise(500, rng)

    res1 = cas_fit(eta, K=2, q=2, r=4, seed=7)
    res2 = cas_fit_streaming(eta, K=2, q=2, r=4, seed=7)

    # V_r is determined up to a rotation Q ∈ O(r); compare via sin-Theta
    sin_theta = sin_theta_F(res1["V_r"], res2["V_r"])
    assert sin_theta < 1e-6, f"cas_fit / cas_fit_streaming disagree: {sin_theta}"


def test_cas_fit_V_r_orthonormal():
    """V_r must be a Stiefel-manifold matrix."""
    rng = np.random.default_rng(1)
    eta = sample_noise(500, rng)
    res = cas_fit(eta, K=2, q=2, r=4, seed=0)
    V = res["V_r"]
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)


@pytest.mark.slow
def test_cas_fit_recovers_oracle_at_large_N():
    """At N=20000 the fitted subspace must be far closer to the exact
    $V_r^C$ than a random subspace is.

    This fit uses the coarse Stage-1 multi-index set (K, q) = (4, 2), at
    which recovery is limited by the Hermite truncation rather than by the
    sample: the paper reports a per-direction angle near 20 degrees on
    Example 1 there, which at r = 4 is sin Theta_F ~ 2 sin(20 deg) ~ 0.68,
    and enriching the interaction order lowers it by roughly an order of
    magnitude. The bound below therefore checks the truncation-limited
    level and guards against regression; it does not certify the accuracy
    reached at the enriched multi-index set.
    """
    rng = np.random.default_rng(0)
    eta = sample_noise(20000, rng)
    res = cas_fit_streaming(eta, K=4, q=2, r=R_ORACLE, seed=0)
    V_star = oracle_subspace(R_ORACLE)
    sin_theta = sin_theta_F(res["V_r"], V_star)
    assert sin_theta < 0.75, \
        f"CAS recovery worse than the (4,2) truncation level: sin Theta = {sin_theta}"

    # A uniformly random rank-r subspace scores about sqrt(r(1 - r/d)) ~ 1.8
    # here; the fit must be far better than that, which is the substantive
    # claim of S5.1.
    V_rand = np.linalg.qr(rng.standard_normal((eta.shape[1], R_ORACLE)))[0]
    sin_rand = sin_theta_F(V_rand, V_star)
    assert sin_theta < 0.5 * sin_rand, \
        f"CAS not clearly better than random: {sin_theta} vs random {sin_rand}"


def test_sin_theta_F_invariant_to_basis_rotation():
    """sin Θ_F must be invariant under right-multiplication by Q ∈ O(r)."""
    rng = np.random.default_rng(2)
    V1 = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    V2 = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    Q = np.linalg.qr(rng.standard_normal((4, 4)))[0]

    s1 = sin_theta_F(V1, V2)
    s2 = sin_theta_F(V1 @ Q, V2)
    np.testing.assert_allclose(s1, s2, atol=1e-12)


def test_sin_theta_F_zero_on_identical_subspaces():
    """sin Θ_F(V, V) = 0."""
    rng = np.random.default_rng(3)
    V = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    np.testing.assert_allclose(sin_theta_F(V, V), 0.0, atol=1e-12)
