"""Componentwise rank-Gaussianization z = Phi^{-1}(F_hat(x)), and
ScalarMarginal: the empirical distribution function with monotone
interpolation and a Gaussian kernel density estimate, cached by the content
hash of the data column.
"""
from __future__ import annotations

from .hermite_score_matching import rank_gaussianize
from .reduced_density import (
    ScalarMarginal,
    rank_transform_through_marginals,
    clear_scalar_marginal_cache,
)

__all__ = [
    "rank_gaussianize",
    "ScalarMarginal",
    "rank_transform_through_marginals",
    "clear_scalar_marginal_cache",
]
