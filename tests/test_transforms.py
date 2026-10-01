"""Tests for ``cas.transforms``.

The transforms module wraps scipy.special.{log_ndtr, ndtri} to compute
logit(Phi(z)) and its inverse stably, without clipping.

Tests cover:
  - Round-trip z -> eta -> z to machine precision in the bulk |z| < 7.
  - Round-trip eta -> z -> eta to machine precision in the bulk |eta| < 14.
  - Tail behaviour: extreme |z| up to 30 gives finite eta (no clipping cap).
  - Tail behaviour: extreme |eta| up to 1000 gives finite z (asymptotic
    Mills' ratio inversion).
  - log_jacobian matches FD of the forward transform.
  - Consistency with the naive (clip-based) implementation in the bulk
    where neither hits the clip threshold.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.special import logit as scipy_logit

from cas.transforms import (
    logit_phi, inv_logit_phi, logit_phi_log_jacobian_z,
)


class TestRoundtrip:
    """Round-trip identity to machine precision."""

    def test_z_to_eta_to_z_bulk(self):
        """For |z| <= 7, round-trip is accurate to ~1e-10."""
        z = np.linspace(-7, 7, 1001)
        eta = logit_phi(z)
        z_back = inv_logit_phi(eta)
        max_err = float(np.abs(z_back - z).max())
        assert max_err < 1e-9, f"max round-trip err = {max_err:.3e}"

    def test_eta_to_z_to_eta_bulk(self):
        """For |eta| <= 14, round-trip is accurate to ~1e-9."""
        eta = np.linspace(-14, 14, 1001)
        z = inv_logit_phi(eta)
        eta_back = logit_phi(z)
        max_err = float(np.abs(eta_back - eta).max())
        assert max_err < 1e-8, f"max round-trip err = {max_err:.3e}"

    def test_eta_to_z_to_eta_moderate_tail(self):
        """For |eta| in [14, 100], round-trip is accurate to ~1e-6."""
        eta = np.linspace(-100, 100, 1001)
        # Filter out the bulk so we test the tail switch.
        mask = np.abs(eta) >= 14
        z = inv_logit_phi(eta[mask])
        eta_back = logit_phi(z)
        max_err = float(np.abs(eta_back - eta[mask]).max())
        # Tail switch uses asymptotics, so slightly looser.
        assert max_err < 1e-5, f"max tail round-trip err = {max_err:.3e}"


class TestExtremeTails:
    """At extreme |z| or |eta|, the function should not return NaN, inf, or
    saturate at a fixed value."""

    def test_no_eta_saturation_large_z(self):
        """eta should grow ~ z^2/2 for large z, not cap at 27.6 like the old clip."""
        z_vals = np.array([7.0, 10.0, 15.0, 20.0, 25.0])
        eta = logit_phi(z_vals)
        # eta should be strictly increasing
        assert np.all(np.diff(eta) > 0), f"eta not monotone: {eta}"
        # eta should exceed the old clip ceiling (27.63)
        assert eta[3] > 30, f"eta at z=20 = {eta[3]}, expected > 30"
        # eta at z = 25 should be ~ 312 (asymptotic 0.5 * 25^2 + log(25 sqrt(2pi)))
        assert 280 < eta[-1] < 350, f"eta at z=25 out of range: {eta[-1]}"

    def test_no_z_saturation_large_eta(self):
        """z should grow ~ sqrt(2 eta) for large eta."""
        eta_vals = np.array([30.0, 50.0, 100.0, 500.0])
        z = inv_logit_phi(eta_vals)
        # z should be strictly increasing
        assert np.all(np.diff(z) > 0)
        # Asymptotic check: z ~ sqrt(2 eta) leading order
        sqrt_2_eta = np.sqrt(2 * eta_vals)
        # Should be within ~10% of asymptotic
        rel_err = float(np.abs(z - sqrt_2_eta).max() / sqrt_2_eta.max())
        assert rel_err < 0.1, f"rel err vs sqrt(2 eta): {rel_err}"

    def test_finite_at_extreme_inputs(self):
        """No NaN or inf at extreme but finite inputs."""
        z_extreme = np.array([-36.0, -20.0, 20.0, 36.0])
        eta = logit_phi(z_extreme)
        assert np.all(np.isfinite(eta)), f"eta not all finite: {eta}"

        eta_extreme = np.array([-700.0, -100.0, 100.0, 700.0])
        z = inv_logit_phi(eta_extreme)
        assert np.all(np.isfinite(z)), f"z not all finite: {z}"


class TestJacobian:
    """log|dz/deta| matches FD of the forward transform."""

    def test_jacobian_vs_fd(self):
        z = np.linspace(-5, 5, 100)
        h = 1e-5
        deta_dz_fd = (logit_phi(z + h) - logit_phi(z - h)) / (2 * h)
        dz_deta_fd = 1.0 / deta_dz_fd
        log_jac_fd = np.log(dz_deta_fd)
        log_jac_analytical = logit_phi_log_jacobian_z(z)
        max_err = float(np.abs(log_jac_analytical - log_jac_fd).max())
        assert max_err < 1e-6, f"jacobian err = {max_err:.3e}"


class TestConsistencyWithNaive:
    """In the bulk where the old (clipped) version doesn't hit the clip,
    new and old should agree to machine precision."""

    def test_logit_phi_matches_naive_bulk(self):
        """For |z| <= 6.5, naive sig = Phi(z) doesn't hit the clip, so the
        naive logit(clip(Phi(z))) should match logit_phi(z)."""
        z = np.linspace(-6.5, 6.5, 100)
        sig_naive = stats.norm.cdf(z)
        sig_naive_clipped = np.clip(sig_naive, 1e-12, 1 - 1e-12)
        eta_naive = np.log(sig_naive_clipped) - np.log1p(-sig_naive_clipped)
        eta_new = logit_phi(z)
        # Beware: naive may lose precision near |z| = 6.5 where Phi ~ 1 - 1e-10.
        max_err = float(np.abs(eta_new - eta_naive).max())
        assert max_err < 1e-6, f"new vs naive max err in bulk: {max_err:.3e}"

    def test_inv_logit_phi_matches_naive_bulk(self):
        """For |eta| <= 13, naive sig = sigmoid(eta) doesn't underflow,
        so ndtri(sig) should match inv_logit_phi(eta)."""
        from scipy.special import ndtri
        eta = np.linspace(-13, 13, 100)
        sig_naive = 1.0 / (1.0 + np.exp(-eta))
        z_naive = ndtri(sig_naive)
        z_new = inv_logit_phi(eta)
        max_err = float(np.abs(z_new - z_naive).max())
        assert max_err < 1e-6, f"new vs naive max err in bulk: {max_err:.3e}"
