#!/usr/bin/env python3
"""Held-out selection of (C_samp, C_curv) on Example 1: reports the pair
selected on the grid {0.3, 0.95, 3, 9.5, 30} x {1e-6, 3.2e-6, 1e-5, 3.2e-5, 1e-4},
the time of the selection, and the coefficient difference from the fit at
(3, 1e-5).

Run: PYTHONPATH=src python scripts/validate_heldout_selection.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from cas import sample_noise
from cas.reduced_density import ReducedDensityModel

GRID_COV = (3.0 / 10.0, 3.0 * 10.0)
GRID_SOB = (1e-5 / 10.0, 1e-5 * 10.0)


def fit_one(scheme, N, seed):
    t0 = time.time()
    model = ReducedDensityModel(
        r=4, K=4, q=2, K_inner=5, q_inner=3,
        use_secondary_rank=False, M_norm=8192, seed=seed,
        inner_ridge_scheme=scheme,
    ).fit(sample_noise(N, np.random.default_rng(20260729 + seed)))
    return model, time.time() - t0


def main():
    print("=== validate heldout ridge selection (Example 1, deployed config) ===",
          flush=True)
    ok = True
    for N in (2000, 20000):
        m_ho, t_ho = fit_one("heldout:c_cov=3,c_sob=1e-5", N, seed=0)
        m_fx, t_fx = fit_one("theoretical:c_cov=3,c_sob=1e-5", N, seed=0)
        cc, cs = m_ho.selected_ridge_constants
        jmin = m_ho.heldout_selection_diag["J_val_min"]
        th_ho, th_fx = m_ho.theta_inner, m_fx.theta_inner
        rel = float(np.linalg.norm(th_ho - th_fx)
                    / max(np.linalg.norm(th_fx), 1e-30))
        in_grid = (GRID_COV[0] <= cc <= GRID_COV[1]
                   and GRID_SOB[0] <= cs <= GRID_SOB[1])
        ok = ok and in_grid and np.isfinite(jmin) and np.isfinite(rel)
        print(f"\n  N={N:>6,}  selected (c_cov, c_sob) = ({cc:.3g}, {cs:.3g})"
              f"  {'[in grid]' if in_grid else '[OUT OF GRID -- FAIL]'}",
              flush=True)
        print(f"           J_val at selection = {jmin:.6f}", flush=True)
        print(f"           fit time: heldout {t_ho:.1f}s vs fixed {t_fx:.1f}s"
              f"  (selection overhead {t_ho - t_fx:+.1f}s)", flush=True)
        print(f"           rel L2 diff of theta_inner vs fixed pair: {rel:.3e}",
              flush=True)

    print("\n  Read: selected values inside the grid and a finite J_val mean",
          flush=True)
    print("  the wired path runs end-to-end on the production constructor.",
          flush=True)
    print("  A small theta difference is expected (the held-out risk is flat",
          flush=True)
    print("  in the scales); a large one would say the selection left the",
          flush=True)
    print("  flat band and deserves a look before the regen.", flush=True)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
