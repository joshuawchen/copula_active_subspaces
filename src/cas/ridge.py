"""Regularized Stage-2 solves. solve_ridge_theoretical implements the
regularizer of Part I, eq:R-combined, with the centering constraint;
solve_ridge_hermitescaled implements R_H2; the other solvers are the
alternatives of the supplement comparisons.
"""
from __future__ import annotations

from .hermite_score_matching import (
    solve_ridge,
    solve_ridge_chaos_block,
    solve_ridge_chaos_block_mom,
    solve_ridge_chaos_block_quantile,
    solve_ridge_chaos_block_sigmoid,
    solve_ridge_trace,
    solve_ridge_trace_hybrid,
    solve_ridge_chaos_block_blend,
    solve_ridge_chaos_eig,
    solve_ridge_full,
    solve_ridge_theoretical,
    solve_ridge_hermitescaled,
)

__all__ = [
    "solve_ridge",
    "solve_ridge_chaos_block",
    "solve_ridge_chaos_block_mom",
    "solve_ridge_chaos_block_quantile",
    "solve_ridge_chaos_block_sigmoid",
    "solve_ridge_trace",
    "solve_ridge_trace_hybrid",
    "solve_ridge_chaos_block_blend",
    "solve_ridge_chaos_eig",
    "solve_ridge_full",
    "solve_ridge_theoretical",
    "solve_ridge_hermitescaled",
]

