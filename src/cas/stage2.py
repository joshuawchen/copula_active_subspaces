"""Streaming assembly of the Hermite score-matching system, used by Stage 1 and
by ReducedDensityModel.fit.
"""
from __future__ import annotations

from .hermite_score_matching import (
    build_A_b,
    build_A_b_streaming,
    build_weighted_A_b_streaming,
    build_psr_A_b_streaming,
    build_psr_b_streaming,
    assemble_T_from_H,
)

__all__ = [
    "build_A_b",
    "build_A_b_streaming",
    "build_weighted_A_b_streaming",
    "build_psr_A_b_streaming",
    "build_psr_b_streaming",
    "assemble_T_from_H",
]
