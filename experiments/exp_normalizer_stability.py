"""Normalizer stability: for one CAS fit per example at N = 50,000, log Z_r and
the log-mass log A at 2^13, 2^15, 2^17 and 2^19 Sobol points under three
independent scrambles. Writes cache/normalizer_stability.json.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cas.config import (R_ORACLE, M_NORM, USE_SECONDARY_RANK, RIDGE_DEFAULT,
                         CACHE_DIR)
from cas import sin_theta_F, ReducedDensityModel
from cas.analytic_oracle import analytic_oracle_subspace
from cas.reduced_density import estimate_log_mass, eval_g_polynomial
from scipy.stats import norm, qmc
from experiments.exp_paper_run import _LAWS, _cache_path, _load

R = R_ORACLE
N_HEAD = 50000
SEED = 0
BUDGETS = [1 << k for k in (13, 15, 17, 19)]
N_SCR = 3


def log_Z_at(model, M, seed):
    U01 = np.clip(qmc.Sobol(d=R, scramble=True, seed=seed).random(M), 1e-12, 1 - 1e-12)
    p = eval_g_polynomial(norm.ppf(U01), model.theta_inner, model.A_inner, model.K_inner)
    mx = float(p.max())
    return mx + float(np.log(np.mean(np.exp(p - mx))))


def one(example):
    sample_fn, _ = _LAWS[example]
    sp = _load(_cache_path(example, "subspace", N_HEAD))
    sel = sp["selected_kq"][str(SEED)]
    rng = np.random.default_rng(7000 + SEED)
    eta_tr = sample_fn(N_HEAD, rng)
    t0 = time.time()
    cas = ReducedDensityModel(
        r=R, K=sel["K1"], q=sel["q1"], even_degree=sel["even_degree"],
        K_inner=sel["K2"], q_inner=sel["q2"], seed=SEED,
        inner_ridge_scheme=RIDGE_DEFAULT, M_norm=M_NORM,
        use_secondary_rank=USE_SECONDARY_RANK).fit(eta_tr)
    fit_s = time.time() - t0
    Vref = analytic_oracle_subspace(example, R)
    st = float(sin_theta_F(cas.V_r, Vref))
    if abs(st - float(sp["CAS"][str(SEED)])) > 1e-8:
        raise SystemExit(f"{example}: Stage-1 angle {st} != cached {sp['CAS'][str(SEED)]}")
    out = {"fit_seconds": round(fit_s, 1), "logZ_production": float(cas.log_Z_norm),
           "logA_production": float(cas.log_mass), "logZ": {}, "logA": {}}
    for M in BUDGETS:
        zs = [log_Z_at(cas, M, 100 + s) for s in range(N_SCR)]
        As = [estimate_log_mass(cas.marginals, cas._log_copula_factor,
                                seed=200 + s, M=M) for s in range(N_SCR)]
        out["logZ"][str(M)] = {"mean": float(np.mean(zs)), "spread": float(np.ptp(zs)),
                               "values": [float(z) for z in zs]}
        out["logA"][str(M)] = {"mean": float(np.mean(As)), "spread": float(np.ptp(As)),
                               "values": [float(a) for a in As]}
        print(f"  {example:15s} M=2^{int(np.log2(M)):2d}  logZ {np.mean(zs):+.5f} "
              f"(spread {np.ptp(zs):.5f})  logA {np.mean(As):+.5f} "
              f"(spread {np.ptp(As):.5f})", flush=True)
    return out


def main():
    res = {"N": N_HEAD, "seed": SEED, "budgets": BUDGETS, "scrambles": N_SCR,
           "M_norm_production": M_NORM}
    for ex in ("banana", "even_fold", "conformal_cube"):
        print(f"=== {ex} ===", flush=True)
        res[ex] = one(ex)
    path = os.path.join(CACHE_DIR, "normalizer_stability.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
