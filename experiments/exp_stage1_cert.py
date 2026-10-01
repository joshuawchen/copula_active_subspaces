"""Stage-1 certificate C_1 = (1/2) tr((I - P_hat) C) against the exact C (Part II,
prop:stage1-cert), from the cached reference eigenpairs and the estimated bases
of the cor44 scan. Writes cache/stage1_cert_{benchmark}.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from cas.analytic_oracle import _cache_path, analytic_oracle_subspace  # noqa: E402

R_DEFAULT = 4
_R_MAX = 8
_N_REF = 1_000_000
_SEED = 20260715


def exact_C(tag: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(C, top eigenvalues, tr C) for the exact copula-score covariance.

    The cached file stores the top-r_max eigenpairs and tr C. The padding
    coordinates have an identically zero copula score, so C has rank at most
    B_SUPPORT = 8 = r_max and the eigenpairs reconstruct it exactly.
    """
    path = _cache_path(tag, _R_MAX, _N_REF, _SEED)
    if not os.path.exists(path):
        analytic_oracle_subspace(tag, R_DEFAULT)      # builds and caches
    z = np.load(path)
    V8 = np.asarray(z["V"], dtype=float)
    lam8 = np.asarray(z["eigvals_top"], dtype=float)
    trC = float(z["trC"])
    resid = trC - float(lam8.sum())
    if abs(resid) > 1e-6 * max(trC, 1.0):
        print(f"  NOTE: tr C exceeds the top-{_R_MAX} eigenvalue sum by "
              f"{resid:.3e}; C is not exactly rank {_R_MAX} and the "
              f"reconstruction drops that remainder")
    return (V8 * lam8) @ V8.T, lam8, trC


def run(benchmark: str, r: int = R_DEFAULT) -> dict:
    C, lam8, trC = exact_C(benchmark)
    d = C.shape[0]
    V_exact = np.asarray(np.load(_cache_path(benchmark, _R_MAX, _N_REF, _SEED))
                         ["V"], dtype=float)[:, :r]
    P_exact = V_exact @ V_exact.T
    rank_price = 0.5 * (trC - float(lam8[:r].sum()))

    scan_path = os.path.join(_REPO, "cache",
                             f"cor44_{benchmark}_scan_K4_q2_r{r}.json")
    scan = json.load(open(scan_path))

    def sin_theta_F(A, B):
        s = np.linalg.svd(A.T @ B, compute_uv=False).clip(0.0, 1.0)
        return float(np.sqrt(max(A.shape[1] - float((s ** 2).sum()), 0.0)))

    cells = []
    for c in scan:
        V_hat = np.asarray(c["V_hat"], dtype=float)
        P_hat = V_hat @ V_hat.T
        C1 = 0.5 * float(np.trace((np.eye(d) - P_hat) @ C))
        subopt = 0.5 * float(np.trace(P_exact @ C) - np.trace(P_hat @ C))
        cells.append({
            "N": int(c["N"]), "seed": int(c["seed"]),
            "C1_exact": C1,
            "Delta_subopt": subopt,
            "rank_price": rank_price,
            "sin_theta_F_vs_exact": sin_theta_F(V_hat, V_exact),
            "trace_bound_C_Lambda": float(c["oracle_KL_bound"]),
            "sin_theta_F_vs_C_Lambda": float(c["sin_theta_F"]),
        })

    out = {
        "benchmark": benchmark, "r": r, "d": d,
        "trC_exact": trC,
        "eigvals_top": [float(x) for x in lam8],
        "rank_price": rank_price,
        "cells": cells,
    }
    path = os.path.join(_REPO, "cache", f"stage1_cert_{benchmark}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)

    print(f"\n=== {benchmark}: Stage-1 certificate against the exact C ===")
    print(f"  tr C = {trC:.4f},  rank price 1/2 sum_(i>r) lambda_i = "
          f"{rank_price:.4f}")
    print(f"  {'N':>7} {'C1(exact)':>11} {'D_subopt':>10} {'sinTh/exact':>12} "
          f"{'1/2tr((I-P)C_L)':>16}")
    for N in sorted({c["N"] for c in cells}):
        rows = [c for c in cells if c["N"] == N]
        m = lambda k: float(np.mean([x[k] for x in rows]))  # noqa: E731
        print(f"  {N:>7} {m('C1_exact'):>11.4f} {m('Delta_subopt'):>10.4f} "
              f"{m('sin_theta_F_vs_exact'):>12.4f} "
              f"{m('trace_bound_C_Lambda'):>16.4f}")
    print(f"  saved {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="all",
                    choices=["banana", "even_fold", "conformal_cube", "all"])
    ap.add_argument("--r", type=int, default=R_DEFAULT)
    args = ap.parse_args()
    benches = (["banana", "even_fold", "conformal_cube"]
               if args.benchmark == "all" else [args.benchmark])
    for b in benches:
        run(b, args.r)


if __name__ == "__main__":
    main()
