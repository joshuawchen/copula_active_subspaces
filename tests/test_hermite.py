"""Tests for ``cas.hermite``.

Verifies the orthonormality of the Hermite basis (against numerical
Gaussian-quadrature integrals) and the dictionary-enumeration constraints
of §A.1 / §3.3.
"""
from __future__ import annotations

import numpy as np
import pytest

from cas.hermite import enumerate_dictionary, hermite_norm


# ---------------------------------------------------------------------------
# Univariate basis
# ---------------------------------------------------------------------------

def test_hermite_norm_constant():
    """h_0 ≡ 1 for any input."""
    z = np.linspace(-3, 3, 10)
    H = hermite_norm(z, max_degree=4)
    np.testing.assert_allclose(H[..., 0], 1.0)


def test_hermite_norm_linear():
    """h_1(z) = z."""
    z = np.linspace(-3, 3, 10)
    H = hermite_norm(z, max_degree=4)
    np.testing.assert_allclose(H[..., 1], z)


def test_hermite_orthonormality():
    """<h_m, h_n>_{γ} = δ_{mn}, by Gauss-Hermite quadrature."""
    # 30 nodes is plenty for degree ≤ 5
    nodes, weights = np.polynomial.hermite_e.hermegauss(30)
    weights = weights / np.sqrt(2 * np.pi)
    H = hermite_norm(nodes, max_degree=5)  # (30, 6)
    gram = (H * weights[:, None]).T @ H
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-10)


# ---------------------------------------------------------------------------
# Dictionary enumeration
# ---------------------------------------------------------------------------

def test_enumerate_dictionary_no_pure_axis_linears():
    """Dictionary must exclude α = e_i for all i (absorbed by score baseline)."""
    Lambda = enumerate_dictionary(d=10, K=4, q=2)
    for supp, deg in Lambda:
        assert not (len(supp) == 1 and deg == (1,)), \
            f"Found pure-axis linear ({supp}, {deg})"


def test_enumerate_dictionary_constraints():
    """Every α has 1 ≤ |α| ≤ K and 1 ≤ ||α||_0 ≤ q."""
    K, q = 4, 2
    Lambda = enumerate_dictionary(d=10, K=K, q=q)
    for supp, deg in Lambda:
        assert 1 <= len(supp) <= q
        assert 1 <= sum(deg) <= K
        # Each component of deg must be ≥ 1 (else it wouldn't be in supp)
        assert all(d >= 1 for d in deg)


def test_enumerate_dictionary_d20_matches_paper():
    """At paper defaults (d=20, K=4, q=2), |Λ| = 1200."""
    Lambda = enumerate_dictionary(d=20, K=4, q=2)
    assert len(Lambda) == 1200


def test_enumerate_dictionary_inner_d4():
    """Inner basis at paper defaults (r=4, K_in=5, q_in=3)."""
    Lambda = enumerate_dictionary(d=4, K=5, q=3)
    assert len(Lambda) == 116
