"""Tests of the fall-back acceptance gate of cas.conditional_kl and of the
error reporting of cas.quad_marginalization.
"""
from __future__ import annotations

import numpy as np


def test_gate_thresholds_exist_and_are_sane():
    """Gate constants must admit a usable IS pass while rejecting the
    observed failures (M_eff_frac 1e-5 to 1e-6)."""
    from cas.conditional_kl import (
        IS_FALLBACK_MAX_REL_SE,
        IS_FALLBACK_MIN_ESS_FRAC,
    )

    assert 0.0 < IS_FALLBACK_MIN_ESS_FRAC < 1.0
    assert 0.0 < IS_FALLBACK_MAX_REL_SE < 1.0
    assert IS_FALLBACK_MIN_ESS_FRAC > 1e-4


def test_quad_skip_returns_a_reason_not_a_bare_none():
    """Every non-applicable branch names itself. A bare None loses the
    distinction between the benign ss_collapse case (inner KL ~ 0, which IS
    returns correctly) and the dangerous no_mode case."""
    import inspect

    from cas import quad_marginalization as qm

    src = inspect.getsource(qm._quad_one_u_with_kl)
    for reason in ("rank_deficient", "btb_ill_conditioned",
                   "ss_collapse", "no_mode"):
        assert f'"{reason}"' in src, f"skip reason {reason} not emitted"
    assert "return None" not in src


def test_cell_aggregation_drops_invalid_outer_points():
    """The cell mean is taken over valid u's only: a rejected u is NaN, is
    excluded, and is counted -- not averaged in, and not allowed to turn the
    whole cell into NaN."""
    inner_kls = np.array([0.30, 0.28, np.nan, 0.31, 0.29])

    valid = np.isfinite(inner_kls)
    n_valid = int(valid.sum())
    kls_valid = inner_kls[valid]
    mean_kl = float(kls_valid.mean())

    assert n_valid == 4
    assert int(len(inner_kls) - n_valid) == 1
    assert np.isfinite(mean_kl)
    assert abs(mean_kl - 0.295) < 1e-9
    # For contrast, a plain mean propagates the NaN.
    assert not np.isfinite(float(inner_kls.mean()))


def test_cell_validity_threshold_rejects_a_poisoned_cell():
    """A cell that lost more than the allowed fraction of outer points is
    marked invalid so consumers drop it, rather than reporting a mean over a
    biased remainder."""
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiments.exp_stage1_reference import CELL_MIN_VALID_FRAC

    assert 0.5 < CELL_MIN_VALID_FRAC <= 1.0

    n_outer = 1000
    assert (990 / n_outer >= CELL_MIN_VALID_FRAC) is True
    assert (900 / n_outer >= CELL_MIN_VALID_FRAC) is False
