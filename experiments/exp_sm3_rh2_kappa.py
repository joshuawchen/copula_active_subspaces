"""R_H2 kappa grid and headline comparison of the Part I supplement
(tab:dw-kappa-grid, tab:stage2-headline): posterior KL on the inference problem
for the unregularized solve, R_H2 at eight values of kappa, and the proposed
regularizer at (C_samp, C_curv) = (3, 10^-5).
Writes cache/sm3_rh2_kappa_bip.json; resumable.
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
    M_NORM, USE_SECONDARY_RANK, CACHE_DIR, BIP_X_STAR,
)
from cas import (
    sample_noise, log_noise_density,
    ReducedDensityModel,
    BIP_A, BIP_XX, posterior_grid, kl,
)

R_BIP = 4
N_VALUES = (100, 500, 2500, 12500, 50000)
KAPPA_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
N_TRIALS_DEFAULT = 50
X_STAR = np.array(BIP_X_STAR)

RIDGE_PROPOSED = "theoretical:c=3,delta=1e-5"
RIDGE_NOREG = "hermitescaled:c=0,alpha=0,delta=1e-12"


def _rh2(kappa: float) -> str:
    return f"hermitescaled:c=0,alpha=0,delta={kappa}"


def _kkey(kappa: float) -> str:
    """Stable cache key for a kappa value: 1e-05, 3e-05, 0.0001, ..., 0.03."""
    return f"{kappa:g}"


CACHE = os.path.join(CACHE_DIR, "sm3_rh2_kappa_bip.json")


def _atomic(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _kw(scheme: str, seed: int = 0) -> dict:
    return dict(
        r=R_BIP, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(R_BIP, Q_INNER_CAP), use_secondary_rank=USE_SECONDARY_RANK,
        M_norm=M_NORM, seed=seed, inner_ridge_scheme=scheme,
    )


def _fit_schemes(N: int) -> dict:
    """Fit all Stage-2 schemes at one N, sharing a single CAS subspace V_r.

    The proposed model is fit first (full CAS), and its V_r is reused (via
    V_r_override) by every other scheme, so the columns differ only in the
    Stage-2 regularizer.
    """
    rng = np.random.default_rng(9000 + N)
    eta_tr = sample_noise(N, rng)

    base = ReducedDensityModel(**_kw(RIDGE_PROPOSED)).fit(eta_tr)
    V_r = base.V_r
    fitted = {"proposed": base, "no_reg":
              ReducedDensityModel(**_kw(RIDGE_NOREG)).fit(eta_tr, V_r_override=V_r)}
    fitted_rh2 = {}
    for kappa in KAPPA_GRID:
        fitted_rh2[_kkey(kappa)] = ReducedDensityModel(
            **_kw(_rh2(kappa))).fit(eta_tr, V_r_override=V_r)
    return {"flat": fitted, "rh2": fitted_rh2}


def _trial_kls(trial_idx: int, models: dict) -> dict:
    """Posterior KL of every scheme on one observation trial."""
    rng_t = np.random.default_rng(20000 + trial_idx)
    eta_obs = sample_noise(1, rng_t)[0]
    y = BIP_A @ X_STAR + eta_obs
    residuals = y - BIP_XX @ BIP_A.T
    p_oracle = posterior_grid(residuals, log_noise_density)

    def _kl(model):
        try:
            return float(kl(p_oracle, posterior_grid(
                residuals, model.evaluate_log_density)))
        except Exception:
            return float("nan")

    out = {name: _kl(m) for name, m in models["flat"].items()}
    out_rh2 = {kk: _kl(m) for kk, m in models["rh2"].items()}
    return {"flat": out, "rh2": out_rh2}


def run(n_trials: int, only_N=None) -> dict:
    if os.path.exists(CACHE):
        data = json.load(open(CACHE))
    else:
        data = {"config": {
            "benchmark": "banana", "rank": R_BIP, "N_values": list(N_VALUES),
            "kappa_grid": list(KAPPA_GRID), "n_trials": n_trials,
            "schemes": {"no_reg": RIDGE_NOREG, "proposed": RIDGE_PROPOSED,
                        "rh2": "hermitescaled:c=0,alpha=0,delta=<kappa>"},
            "shared_subspace": "CAS V_r reused across schemes per N",
        }, "cells": {}}
    Ns = [only_N] if only_N else list(N_VALUES)
    t0 = time.time()
    for N in Ns:
        cell = data["cells"].setdefault(str(N), {})
        cell.setdefault("no_reg", [])
        cell.setdefault("proposed", [])
        cell.setdefault("rh2", {})
        for kappa in KAPPA_GRID:
            cell["rh2"].setdefault(_kkey(kappa), [])
        done = len(cell["no_reg"])
        if done >= n_trials:
            print(f"  N={N}: {done}/{n_trials} (complete)", flush=True)
            continue
        models = _fit_schemes(N)
        print(f"  N={N}: fitted schemes [{time.time()-t0:.0f}s], "
              f"trials {done}->{n_trials}", flush=True)
        for t in range(done, n_trials):
            res = _trial_kls(t, models)
            cell["no_reg"].append(res["flat"]["no_reg"])
            cell["proposed"].append(res["flat"]["proposed"])
            for kk, v in res["rh2"].items():
                cell["rh2"][kk].append(v)
            _atomic(CACHE, data)
        print(f"  N={N}: done {n_trials} trials [{time.time()-t0:.0f}s]", flush=True)
    return data


def _mean(a):
    a = np.array([v for v in a if v is not None and not np.isnan(v)])
    return float(a.mean()) if len(a) else float("nan")


def summarize(data: dict) -> None:
    cells = data["cells"]
    print("\n=== tab:dw-kappa-grid (mean BIP posterior KL, r=4) ===")
    head = "  N      |" + "".join(f" {k:>7g}" for k in KAPPA_GRID)
    print(head)
    oracle = {}  # N -> (best_kappa, best_mean)
    for N in N_VALUES:
        c = cells.get(str(N))
        if not c or not c["no_reg"]:
            continue
        means = [_mean(c["rh2"][_kkey(k)]) for k in KAPPA_GRID]
        j = int(np.nanargmin(means))
        oracle[N] = (KAPPA_GRID[j], means[j])
        row = f"  {N:<6} |" + "".join(
            f" {m:7.2f}" + ("*" if i == j else " ")[:0] for i, m in enumerate(means))
        # mark the row-min with a trailing tag
        row = f"  {N:<6} |" + "".join(
            (f"[{m:6.2f}]" if i == j else f" {m:6.2f} ") for i, m in enumerate(means))
        print(row)
    print("  ([.] = oracle-best kappa at that N)")

    print("\n=== tab:stage2-headline (BIP posterior KL, r=4) ===")
    print(f"  {'N':<7} {'no reg':>8} {'R_H2 (oracle k)':>18} {'proposed':>10}   winner")
    for N in N_VALUES:
        c = cells.get(str(N))
        if not c or not c["no_reg"]:
            continue
        nr = _mean(c["no_reg"])
        pr = _mean(c["proposed"])
        bk, bm = oracle.get(N, (float("nan"), float("nan")))
        cand = {"no reg": nr, "R_H2": bm, "proposed": pr}
        win = min(cand, key=lambda k: cand[k])
        print(f"  {N:<7} {nr:>8.2f} {bm:>10.2f} (k={bk:g}) {pr:>10.2f}   {win}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--only-N", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke: implies a small --n-trials if not set")
    args = ap.parse_args()
    n_trials = args.n_trials
    if args.smoke and n_trials == N_TRIALS_DEFAULT:
        n_trials = 3
    print(f"=== SM3 R_H2 kappa-sweep (BIP r={R_BIP}, {n_trials} trials) ===",
          flush=True)
    summarize(run(n_trials, args.only_N))


if __name__ == "__main__":
    main()
