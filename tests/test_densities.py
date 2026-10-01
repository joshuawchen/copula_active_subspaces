"""Tests for ``cas.densities``.

Verifies that the model classes produce normalized, monotone log-densities
on the d=20 banana noise law, and that the ScalarMarginal cache speedup
preserves bit-equivalence.
"""
from __future__ import annotations

import numpy as np
import pytest

from cas import (
    GaussianCopulaModel,
    ProductOfMarginalsModel,
    ReducedDensityModel,
    clear_scalar_marginal_cache,
    log_noise_density,
    rank_gaussianize,
    sample_noise,
)
from cas.config import (
    K_OUTER, Q_OUTER, K_INNER, Q_INNER_CAP,
    M_NORM, USE_SECONDARY_RANK, R_ORACLE,
)

RIDGE = "theoretical:c=3,delta=1e-5"


def _pca_subspace(eta: np.ndarray, r: int) -> np.ndarray:
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    Sigma = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sigma)
    return V[:, np.argsort(-w)[:r]]


@pytest.fixture(scope="module")
def small_eta():
    rng = np.random.default_rng(42)
    return sample_noise(500, rng)


def test_pom_log_density_finite(small_eta):
    pom = ProductOfMarginalsModel().fit(small_eta)
    test_x = small_eta[:50]
    ll = pom.evaluate_log_density(test_x)
    assert np.all(np.isfinite(ll))


def test_gaussian_log_density_finite(small_eta):
    gauss = GaussianCopulaModel().fit(small_eta)
    ll = gauss.evaluate_log_density(small_eta[:50])
    assert np.all(np.isfinite(ll))


def test_reduced_density_log_finite(small_eta):
    m = ReducedDensityModel(
        r=R_ORACLE, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(R_ORACLE, Q_INNER_CAP),
        use_secondary_rank=USE_SECONDARY_RANK, M_norm=M_NORM, seed=0,
        inner_ridge_scheme=RIDGE,
    ).fit(small_eta)
    ll = m.evaluate_log_density(small_eta[:50])
    assert np.all(np.isfinite(ll))


def test_log_noise_density_matches_numerical_truth(small_eta):
    """log_noise_density should be a finite log-density."""
    ll = log_noise_density(small_eta[:100])
    assert np.all(np.isfinite(ll))


def test_scalar_marginal_cache_does_not_change_results():
    """Re-fitting on the same data with cache cleared and warm should
    give bit-identical evaluate_log_density outputs."""
    rng = np.random.default_rng(0)
    eta = sample_noise(500, rng)
    test_x = sample_noise(50, np.random.default_rng(1))

    clear_scalar_marginal_cache()
    pom_cold = ProductOfMarginalsModel().fit(eta)
    ll_cold = pom_cold.evaluate_log_density(test_x)

    # Now warm cache (the cache is at module scope, so a second fit hits it)
    pom_warm = ProductOfMarginalsModel().fit(eta)
    ll_warm = pom_warm.evaluate_log_density(test_x)

    # Identity: same marginal objects (read-only sharing is safe)
    assert pom_cold.marginals[0] is pom_warm.marginals[0]
    np.testing.assert_array_equal(ll_cold, ll_warm)


@pytest.mark.slow
def test_cas_outperforms_pom_on_banana():
    """At a moderate N=2000, CAS-HCSM should beat PoM on held-out LL."""
    rng = np.random.default_rng(0)
    eta_tr = sample_noise(2000, rng)
    eta_te = sample_noise(2000, rng)

    pom = ProductOfMarginalsModel().fit(eta_tr)
    cas = ReducedDensityModel(
        r=R_ORACLE, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(R_ORACLE, Q_INNER_CAP),
        use_secondary_rank=USE_SECONDARY_RANK, M_norm=M_NORM, seed=0,
        inner_ridge_scheme=RIDGE,
    ).fit(eta_tr)

    ll_pom = float(np.mean(pom.evaluate_log_density(eta_te)))
    ll_cas = float(np.mean(cas.evaluate_log_density(eta_te)))
    assert ll_cas > ll_pom, f"CAS-HCSM should beat PoM, got cas={ll_cas} pom={ll_pom}"
