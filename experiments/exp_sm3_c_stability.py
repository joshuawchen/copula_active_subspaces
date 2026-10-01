"""Stage-2 coefficient stability of the Part I supplement (tab:s2reg-ablations):
paired held-out log-likelihood of R at C in {1, 3, 10} against R_H2 at
kappa = 10^-3, for r in {2, 3, 4, 6, 8} and N in {500, 1000, 2000}.
Writes cache/sm3_c_stability_banana.json; resumable.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from cas.config import (
    K_OUTER, Q_OUTER, K_INNER, Q_INNER_CAP,
    M_NORM, USE_SECONDARY_RANK, CACHE_DIR,
)
from cas import sample_noise, ReducedDensityModel

R_VALUES = (2, 3, 4, 6, 8)
N_VALUES = (500, 1000, 2000)
N_SEEDS_DEFAULT = 20
SEED_BASE = 7000                 # matches Sec 5.2 / SM1-banana seeding
SPLIT = 0.8                      # 80/20 train/test
C_VALUES = (1, 3, 10)
RIDGE_RH2 = "hermitescaled:c=0,alpha=0,delta=1e-3"   # R_H2 at kappa = 1e-3


def RIDGE_R(c: int) -> str:
    return f"theoretical:c={c},delta=1e-5"


CACHE = os.path.join(CACHE_DIR, "sm3_c_stability_banana.json")
WON_TOL = 0.05                   # cell counts as won / lost at this margin


def _atomic(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _kw(scheme: str, r: int, seed: int) -> dict:
    return dict(
        r=r, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(r, Q_INNER_CAP), use_secondary_rank=USE_SECONDARY_RANK,
        M_norm=M_NORM, seed=seed, inner_ridge_scheme=scheme,
    )


def _cell(r: int, N: int, seed: int) -> dict:
    """One (r,N,seed): R_H2 (full CAS) + R c in {1,3,10} on the shared V_r."""
    rng = np.random.default_rng(SEED_BASE + seed)
    eta = sample_noise(N, rng)
    n_tr = int(SPLIT * N)
    tr, te = eta[:n_tr], eta[n_tr:]
    base = ReducedDensityModel(**_kw(RIDGE_RH2, r, seed)).fit(tr)
    V_r = base.V_r
    out = {"rh2": float(np.mean(base.evaluate_log_density(te)))}
    for c in C_VALUES:
        m = ReducedDensityModel(**_kw(RIDGE_R(c), r, seed)).fit(
            tr, V_r_override=V_r)
        out[f"c{c}"] = float(np.mean(m.evaluate_log_density(te)))
    return out


def run(n_seeds: int, only_r=None, only_N=None) -> dict:
    if os.path.exists(CACHE):
        data = json.load(open(CACHE))
    else:
        data = {"config": {
            "benchmark": "banana", "r_values": list(R_VALUES),
            "N_values": list(N_VALUES), "n_seeds": n_seeds, "split": SPLIT,
            "seed_base": SEED_BASE, "c_values": list(C_VALUES),
            "rh2_scheme": RIDGE_RH2,
            "r_scheme": "theoretical:c=<c>,delta=1e-5",
            "shared_subspace": "CAS V_r reused across schemes per (r,N,seed)",
        }, "cells": {}}
    rs = [only_r] if only_r else list(R_VALUES)
    Ns = [only_N] if only_N else list(N_VALUES)
    t0 = time.time()
    for r in rs:
        data["cells"].setdefault(str(r), {})
        for N in Ns:
            cell = data["cells"][str(r)].setdefault(
                str(N), {"rh2": [], **{f"c{c}": [] for c in C_VALUES}})
            done = len(cell["rh2"])
            for seed in range(done, n_seeds):
                res = _cell(r, N, seed)
                cell["rh2"].append(res["rh2"])
                for c in C_VALUES:
                    cell[f"c{c}"].append(res[f"c{c}"])
                _atomic(CACHE, data)
            print(f"  r={r} N={N}: {len(cell['rh2'])}/{n_seeds} seeds "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return data


def _cell_gap(cell: dict, c: int) -> float:
    """Per-cell mean paired gap (R c minus R_H2), over seeds."""
    a = np.array(cell[f"c{c}"]) - np.array(cell["rh2"])
    return float(a.mean()) if len(a) else float("nan")


def summarize(data: dict) -> None:
    cells = data["cells"]
    print("\n=== tab:s2reg-ablations (Stage-2 c-stability vs R_H2) ===")
    print(f"  {'scheme':<14} {'mean':>7} {'min cell':>9} {'max cell':>9} "
          f"{'lost':>5} {'won':>4}   (min@ / max@)")
    for c in C_VALUES:
        gaps = {}
        for r in R_VALUES:
            for N in N_VALUES:
                cell = cells.get(str(r), {}).get(str(N))
                if cell and cell["rh2"]:
                    gaps[(r, N)] = _cell_gap(cell, c)
        if not gaps:
            continue
        vals = np.array(list(gaps.values()))
        keys = list(gaps.keys())
        imin = int(np.argmin(vals))
        imax = int(np.argmax(vals))
        lost = int(np.sum(vals < -WON_TOL))
        won = int(np.sum(vals > WON_TOL))
        tag = " (proposed)" if c == 3 else ""
        print(f"  R c={c:<2}{tag:<9} {vals.mean():>+7.3f} {vals.min():>+9.3f} "
              f"{vals.max():>+9.3f} {lost:>5} {won:>4}   "
              f"(r={keys[imin][0]},N={keys[imin][1]} / "
              f"r={keys[imax][0]},N={keys[imax][1]})")
    # how often does every c beat R_H2 on average across cells?
    print("\n  (per-cell mean gaps, for the reading prose)")
    for c in C_VALUES:
        row = []
        for r in R_VALUES:
            for N in N_VALUES:
                cell = cells.get(str(r), {}).get(str(N))
                if cell and cell["rh2"]:
                    row.append(f"({r},{N}):{_cell_gap(cell, c):+.2f}")
        print(f"    c={c}: " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--only-r", type=int, default=None)
    ap.add_argument("--only-N", type=int, default=None)
    args = ap.parse_args()
    print(f"=== SM3 Stage-2 c-stability (banana test-LL, {args.n_seeds} seeds) ===",
          flush=True)
    summarize(run(args.n_seeds, args.only_r, args.only_N))


if __name__ == "__main__":
    main()
