"""Orthogonal latent mixing shared by Examples 1-3: a rotation Q drawn from the
Haar measure on O(B_SUPPORT), B_SUPPORT = 8, acting on the first B_SUPPORT
coordinates and as the identity on the rest.
"""
from __future__ import annotations

import numpy as np

B_SUPPORT: int = 8          # coordinates the rotation touches = the copula rank
ROT_SEED: int = 20260714


def block_rotation(d: int, b: int = B_SUPPORT, seed: int = ROT_SEED) -> np.ndarray:
    """Haar-random rotation acting on the first b coordinates, identity elsewhere.

    Sign-fixed so the QR factorization is unique and the law is reproducible
    across sessions and machines.
    """
    rng = np.random.default_rng(seed)
    Qb, R = np.linalg.qr(rng.standard_normal((b, b)))
    Qb = Qb * np.sign(np.diag(R))[None, :]
    Q = np.eye(d)
    Q[:b, :b] = Qb
    return Q
