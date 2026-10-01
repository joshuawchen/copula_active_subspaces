"""Tests for ``cas.ranking``.

Verifies that rank-Gaussianization produces approximately Gaussian marginals
and that ``ScalarMarginal`` evaluates a normalized density.
"""
from __future__ import annotations

import numpy as np
import pytest

from cas.ranking import (
    ScalarMarginal,
    clear_scalar_marginal_cache,
    rank_gaussianize,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_rank_gaussianize_marginal_mean_var(rng):
    """After rank-Gaussianization each column should be ≈ N(0, 1)."""
    X = rng.uniform(-5, 5, size=(5000, 4))
    Z = rank_gaussianize(X)
    if isinstance(Z, tuple):
        Z = Z[0]
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=0.05)


def test_rank_gaussianize_preserves_rank_order(rng):
    """The rank of each column must be preserved under rank-Gaussianization."""
    X = rng.standard_normal((1000, 3))
    Z = rank_gaussianize(X)
    if isinstance(Z, tuple):
        Z = Z[0]
    for i in range(X.shape[1]):
        assert np.array_equal(np.argsort(X[:, i]), np.argsort(Z[:, i]))


def test_scalar_marginal_density_normalized(rng):
    """∫ f̂(x) dx ≈ 1 over the support."""
    x = rng.standard_normal(2000)
    m = ScalarMarginal(x)
    grid = np.linspace(x.min() - 3, x.max() + 3, 2000)
    pdf = np.exp(m.log_pdf(grid))
    integral = np.trapezoid(pdf, grid)
    assert 0.95 < integral < 1.05


def test_scalar_marginal_log_pdf_is_kde_everywhere(rng):
    """log_pdf equals the log of the Gaussian KDE at every point, including
    far beyond the sample range, so the marginal is the KDE the paper
    describes and its tails are Gaussian, hence integrable. Beyond the
    tabulated grid the value is the KDE's own log-density, evaluated in log
    space, so there is no floor."""
    # A small sample has a wide bandwidth, so the KDE stays representable
    # well past the +/-6 sigma padding of the tabulated grid, which is
    # where a constant tail and a Gaussian one differ.
    x = rng.standard_normal(30)
    m = ScalarMarginal(x)
    grid = np.concatenate([np.linspace(x.min(), x.max(), 50),
                           np.linspace(x.max() + 7.0, x.max() + 12.0, 25),
                           np.linspace(x.min() - 12.0, x.min() - 7.0, 25)])
    logp = m.log_pdf(grid)
    logk = m.kde.logpdf(grid)
    # Interior: interpolation error only (PCHIP on a 5000-point grid).
    # Exterior: exact.
    np.testing.assert_allclose(logp[:50], logk[:50], atol=1e-4)
    np.testing.assert_array_equal(logp[50:], logk[50:])
    # And the tail decays: the last exterior point on each side is far
    # below the value nearer the sample, which a constant tail would not be.
    assert logp[74] < logp[50] - 50.0
    assert logp[75] < logp[99] - 50.0
    # Whole-line mass: integrating exp(log_pdf) over a range that extends far
    # beyond the old grid padding gives one; a constant tail would add mass
    # proportional to the range.
    wide = np.linspace(x.min() - 40.0, x.max() + 40.0, 40000)
    integral = np.trapezoid(np.exp(m.log_pdf(wide)), wide)
    assert abs(integral - 1.0) < 1e-3


def test_estimate_log_mass_is_zero_for_a_normalized_factor(rng):
    """With a copula factor identically one, the composed estimate is the
    product of the kernel marginals, which integrates to one exactly, so
    the estimated log-mass is zero up to quasi-Monte Carlo error."""
    from cas.reduced_density import estimate_log_mass
    X = rng.standard_normal((300, 3))
    marginals = [ScalarMarginal(X[:, i]) for i in range(3)]
    logA = estimate_log_mass(marginals, lambda Xq: np.zeros(len(Xq)), seed=0,
                             M=4096)
    assert abs(logA) < 1e-12


def test_estimate_log_mass_recovers_a_known_mass(rng):
    """A copula factor equal to a constant c multiplies the mass by c."""
    from cas.reduced_density import estimate_log_mass
    X = rng.standard_normal((300, 2))
    marginals = [ScalarMarginal(X[:, i]) for i in range(2)]
    logA = estimate_log_mass(marginals, lambda Xq: np.full(len(Xq), np.log(2.5)),
                             seed=1, M=4096)
    assert abs(logA - np.log(2.5)) < 1e-9


def test_scalar_marginal_cdf_monotonic(rng):
    """F̂ must be non-decreasing."""
    x = rng.standard_normal(500)
    m = ScalarMarginal(x)
    grid = np.linspace(-5, 5, 200)
    cdf = m.cdf(grid)
    diffs = np.diff(cdf)
    assert np.all(diffs >= -1e-12), "CDF is not monotone non-decreasing"


def test_scalar_marginal_cache_identity(rng):
    """Two ScalarMarginals built from identical data should be the *same object*."""
    clear_scalar_marginal_cache()
    x = rng.standard_normal(500)

    # The cache lives at module level inside reduced_density; check via the
    # public _build_marginal_cached helper indirectly through model classes
    from cas.densities import ProductOfMarginalsModel

    X = np.tile(x[:, None], (1, 3))   # 3 identical columns
    pom = ProductOfMarginalsModel().fit(X)
    # Same column data → same cached object
    assert pom.marginals[0] is pom.marginals[1]
    assert pom.marginals[0] is pom.marginals[2]


def test_rank_gaussianize_d20_banana():
    """End-to-end: sample from the banana noise law, rank-Gaussianize, and
    confirm the result is approximately Gaussian (KS-test slack)."""
    from cas import sample_noise

    rng = np.random.default_rng(0)
    eta = sample_noise(2000, rng)
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    # Each column's z-score histogram should match the standard normal
    # within sample-size tolerance. Just check the first-two moments.
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=0.05)
