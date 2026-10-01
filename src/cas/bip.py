"""Bayesian inference problem of Part I, Section 5: y = A x + eta with a
two-dimensional parameter; posteriors are evaluated on an 80 x 80 grid over
[-6, 6]^2.
"""
from __future__ import annotations
import numpy as np

from .config import (BIP_SIGMA_PRIOR, BIP_ALPHA_SIGNAL,
                    BIP_GRID_MIN, BIP_GRID_MAX, BIP_GRID_N)
from .noise import L_CHOL


# Forward map (§5.3): observe the two conformal INPUT directions (coords 0,1
# of W), unit-normalized and rotated through L. The noise carries z^2 with
# z = W0 + i W1, so inverting these inputs is 2-to-1: the posterior over x is
# bimodal (the two preimages share a radius, hence equal Gaussian weight). The
# Cas reduced model recovers both modes; a Gaussian copula collapses to one.
_BIP_LIN = L_CHOL[:, 0:2] / np.linalg.norm(L_CHOL[:, 0:2], axis=0)
BIP_A = BIP_ALPHA_SIGNAL * _BIP_LIN          # (d, 2)

# Posterior grid (precomputed)
_bip_x         = np.linspace(BIP_GRID_MIN, BIP_GRID_MAX, BIP_GRID_N)
BIP_X1, BIP_X2 = np.meshgrid(_bip_x, _bip_x)
BIP_XX         = np.stack([BIP_X1.ravel(), BIP_X2.ravel()], axis=1)  # (G^2, 2)
BIP_DX2        = (_bip_x[1] - _bip_x[0]) ** 2


def log_prior(X):
    """log p_0(x) for x ~ N(0, BIP_SIGMA_PRIOR^2 I_2)."""
    return (-0.5 * np.sum(X ** 2, axis=1) / BIP_SIGMA_PRIOR ** 2
            - np.log(2 * np.pi * BIP_SIGMA_PRIOR ** 2))


def log_gauss(eta, mu, Sigma):
    """Log density of multivariate N(mu, Sigma) at rows of eta."""
    d = eta.shape[-1]
    _, logdet = np.linalg.slogdet(Sigma)
    inv_S = np.linalg.inv(Sigma)
    diff  = eta - mu
    quad  = np.einsum('...i,ij,...j->...', diff, inv_S, diff)
    return -0.5 * quad - 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)


def posterior_grid(residuals, log_noise_fn):
    """Posterior on the BIP grid given residuals = y - A * BIP_XX and a log-density.

    residuals: (G^2, d). log_noise_fn: callable on (G^2, d) -> (G^2,).
    """
    lp = log_prior(BIP_XX) + log_noise_fn(residuals)
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum() * BIP_DX2
    return p


def kl(p, q):
    """KL(p || q) on the BIP grid; both p, q have shape (G^2,)."""
    eps = 1e-15
    m = p > eps
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], eps, None))) * BIP_DX2)


def draw_y(rng, x_star, sample_noise_fn):
    """One observation y = A x_star + eta, eta ~ noise."""
    eta = sample_noise_fn(1, rng)[0]
    return BIP_A @ x_star + eta
