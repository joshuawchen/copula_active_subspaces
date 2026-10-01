"""Reference subspace for the experiments: deployment_oracle_subspace returns
the V_r^C of cas.analytic_oracle.
"""
from __future__ import annotations

import numpy as np

from .analytic_oracle import analytic_oracle_subspace


def deployment_oracle_subspace(tag: str, r: int) -> np.ndarray:
    """Top-r columns of the exact copula active subspace V_r^C for `tag`.

    Delegates to cas.analytic_oracle.analytic_oracle_subspace, which builds and
    caches the exact copula-score covariance from the analytic score. The name
    and signature are retained so existing callers (oracle_subspace_* in the
    per-example modules) are unchanged.
    """
    return analytic_oracle_subspace(tag, r)
