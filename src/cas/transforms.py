"""Stable componentwise maps between z (standard-normal marginals) and
eta = logit(Phi(z)), without clipping:

    logit_phi(z)                  log_ndtr(z) - log_ndtr(-z)
    inv_logit_phi(eta)            Phi^{-1}(sigmoid(eta)), reflected for large
                                  |eta|, Mills-ratio asymptote beyond 700
    logit_phi_log_jacobian_z(z)   log |dz / deta|
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtri


_LOG_2PI = float(np.log(2.0 * np.pi))


def logit_phi(z: np.ndarray) -> np.ndarray:
    """Componentwise eta = logit(Phi(z)).

    Stable for |z| up to ~37 (where log_ndtr loses precision); beyond that
    returns +/-inf which is the correct limit.
    """
    z = np.asarray(z, dtype=np.float64)
    return log_ndtr(z) - log_ndtr(-z)


def inv_logit_phi(eta: np.ndarray) -> np.ndarray:
    """Componentwise z = Phi^{-1}(sigmoid(eta)).

    Stable everywhere via a bulk/tail switch.
    """
    eta = np.asarray(eta, dtype=np.float64)
    out = np.empty_like(eta)

    # Bulk: |eta| < 14, sigmoid(eta) far from 0/1 so ndtri is precise.
    bulk = np.abs(eta) < 14.0
    if bulk.any():
        sig = 1.0 / (1.0 + np.exp(-eta[bulk]))
        out[bulk] = ndtri(sig)

    # Moderate tail: 14 <= |eta| < 700.
    # For eta > 0: sigmoid(eta) -> 1, so use z = -ndtri(sigmoid(-eta))
    # since sigmoid(-eta) is a small positive number that ndtri handles well.
    mid_pos = (eta >= 14.0) & (eta < 700.0)
    if mid_pos.any():
        sig_neg = 1.0 / (1.0 + np.exp(eta[mid_pos]))
        out[mid_pos] = -ndtri(sig_neg)
    mid_neg = (eta <= -14.0) & (eta > -700.0)
    if mid_neg.any():
        sig_pos = 1.0 / (1.0 + np.exp(-eta[mid_neg]))
        # sig_pos here is a small positive number (sigmoid of large negative eta)
        out[mid_neg] = ndtri(sig_pos)

    # Extreme tail: |eta| >= 700; even exp(-|eta|) underflows.
    # Use Mills' ratio asymptotic:
    #   z = sqrt(2|eta|) - log(4 pi |eta|) / (2 sqrt(2|eta|))
    extreme_pos = eta >= 700.0
    if extreme_pos.any():
        e = eta[extreme_pos]
        sqrt_2e = np.sqrt(2.0 * e)
        out[extreme_pos] = sqrt_2e - np.log(4.0 * np.pi * e) / (2.0 * sqrt_2e)
    extreme_neg = eta <= -700.0
    if extreme_neg.any():
        e = -eta[extreme_neg]
        sqrt_2e = np.sqrt(2.0 * e)
        out[extreme_neg] = -(sqrt_2e - np.log(4.0 * np.pi * e) / (2.0 * sqrt_2e))

    return out


def logit_phi_log_jacobian_z(z: np.ndarray) -> np.ndarray:
    """log |dz/deta| as a function of z, where eta = logit(Phi(z)).

    Identity:
        deta/dz = phi(z) / [Phi(z) (1 - Phi(z))]
    so log|dz/deta| = log Phi(z) + log(1 - Phi(z)) - log phi(z)
                    = log_ndtr(z) + log_ndtr(-z) + 0.5 z^2 + 0.5 log(2 pi).
    Stable everywhere log_ndtr is.
    """
    z = np.asarray(z, dtype=np.float64)
    return log_ndtr(z) + log_ndtr(-z) + 0.5 * z * z + 0.5 * _LOG_2PI
