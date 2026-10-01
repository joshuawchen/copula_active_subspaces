"""Stage 1 of Part I: rank-Gaussianize, fit the Hermite copula score by score
matching, form the score second-moment matrix and return its top-r
eigenvectors. cas_fit holds the features in memory; cas_fit_streaming builds
them in blocks.
"""
from __future__ import annotations

# Re-export numerics from the well-tested implementation. The legacy module
# carries the verified dense per-axis Phi rewrite (~18-30× speedup) and the
# block-streamed Gram assembly used by Algorithm 1.
from .hermite_score_matching import (
    cas_fit,
    cas_fit_streaming,
    sin_theta_F,
)

__all__ = ["cas_fit", "cas_fit_streaming", "sin_theta_F"]
