"""Normalized probabilists' Hermite polynomials h_n = He_n / sqrt(n!), their
tensor products H_alpha, and the multi-index set Lambda_{K,q} (total degree at
most K, at most q active coordinates, no linear modes), stored in sparse form.
"""
from __future__ import annotations
import math
from itertools import combinations
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Univariate evaluation
# ---------------------------------------------------------------------------

def hermite_norm(z: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Evaluate normalized probabilists' Hermite polynomials ``h_0, ..., h_{max_degree}``
    at every point of ``z``.

    Implements (A.1.1)–(A.1.3): builds the unnormalized He_n via the three-term
    recurrence, then divides by sqrt(n!) componentwise.

    Parameters
    ----------
    z : array_like, any shape
        Evaluation points.
    max_degree : int
        Highest degree to evaluate.

    Returns
    -------
    H : ndarray, shape z.shape + (max_degree + 1,)
        ``H[..., n]`` is ``h_n(z)``.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape + (max_degree + 1,), dtype=float)
    out[..., 0] = 1.0
    if max_degree >= 1:
        out[..., 1] = z
        # He_n recurrence (unnormalized); normalize after the loop
        for n in range(1, max_degree):
            out[..., n + 1] = z * out[..., n] - n * out[..., n - 1]
    norms = np.array([math.sqrt(math.factorial(n))
                      for n in range(max_degree + 1)])
    return out / norms


# ---------------------------------------------------------------------------
# Dictionary enumeration
# ---------------------------------------------------------------------------

def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield all compositions of ``total`` into ``parts`` positive integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def enumerate_dictionary(
    d: int,
    K: int,
    q: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Enumerate the truncated tensor-product Hermite dictionary Λ of §3.3.

    Each multi-index α is returned in sparse form ``(supp, deg)`` where
    ``supp = (i_1, ..., i_m)`` is the strictly increasing list of nonzero
    coordinates and ``deg = (α_{i_1}, ..., α_{i_m})`` are the corresponding
    positive degrees, so |supp| = ‖α‖₀ and sum(deg) = |α|.

    The dictionary excludes:
      * the constant α = 0 (orthogonal to score baseline);
      * pure-axis linears α = eᵢ (absorbed by ``-z`` shift; see §3.3 Eq 3.4).

    Parameters
    ----------
    d : int
        Ambient dimension.
    K : int
        Total-degree cap, |α| ≤ K.
    q : int
        Interaction-order cap, ‖α‖₀ ≤ q.

    Returns
    -------
    Λ : list of (supp, deg) tuples
        Sparse dictionary of size |Λ|.
    """
    Lambda: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for m in range(1, q + 1):
        for supp in combinations(range(d), m):
            for total in range(m, K + 1):  # need total ≥ m (each deg ≥ 1)
                for deg in _compositions(total, m):
                    # Exclude pure-axis linears α = eᵢ (m == 1 and deg == (1,))
                    if m == 1 and deg == (1,):
                        continue
                    Lambda.append((supp, deg))
    return Lambda
