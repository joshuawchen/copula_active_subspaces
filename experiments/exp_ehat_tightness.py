#!/usr/bin/env python3
"""Tail-sum accuracy of the Part I supplement (tab:ehat-tightness): per seed at
N = 50,000, the tail sum hatE_r at (K1, q1) = (4, 2), its error against the
reference E_r(C_ref) and the two bounds of eq:Ehat-weyl.
Writes cache/ehat_tightness_{benchmark}.json.

Usage: python3 experiments/exp_ehat_tightness.py --benchmark all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from cas import sample_noise
from cas.even_fold import sample_even_fold_noise
from cas.conformal_cube import sample_conformal_cube_noise
from cas.hermite_score_matching import (rank_gaussianize, enumerate_dictionary,
                              build_A_b_streaming, assemble_T_from_H)

SMP = {"banana": sample_noise, "even_fold": sample_even_fold_noise,
       "conformal_cube": sample_conformal_cube_noise}
K, Q, R, KAPPA, SEED0 = 4, 2, 4, 1e-4, 20260701


def fit_theta(Z):
    d = Z.shape[1]
    A = enumerate_dictionary(d, K, Q)
    Am, b, _, _, H = build_A_b_streaming(Z, A, K, keep_Phi=False)
    td = np.array([sum(dg) for (_, dg) in A], float)
    lam = float(eigsh(Am, k=1, which="LM", return_eigenvectors=False)[0])
    Am[np.diag_indices_from(Am)] += (KAPPA * lam) * td ** 2
    return np.linalg.solve(Am, b), A, H


def cov_of(H, A, th, d, M):
    T = assemble_T_from_H(H, A, th, d)
    C = T @ T.T / M
    return 0.5 * (C + C.T)


def run_benchmark(tag, nref, meval, N, seeds, out_dir):
    smp = SMP[tag]
    t0 = time.time()
    Zev = rank_gaussianize(smp(meval, np.random.default_rng(SEED0 + 999)))
    d = Zev.shape[1]
    Aref = enumerate_dictionary(d, K, Q)
    _, _, _, _, Hev = build_A_b_streaming(Zev, Aref, K, keep_Phi=False)
    th_ref, _, _ = fit_theta(rank_gaussianize(smp(nref, np.random.default_rng(SEED0))))
    C_Lam = cov_of(Hev, Aref, th_ref, d, meval)
    wL = np.sort(np.linalg.eigvalsh(C_Lam))[::-1]
    E_r = float(wL[R:].sum())
    print(f"[{tag}] d={d} nref={nref} meval={meval} trC_Lam={np.trace(C_Lam):.4f} "
          f"E_r={E_r:.4f} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for s in range(seeds):
        eta = smp(N, np.random.default_rng(SEED0 + 1009 * s + 7 * N))
        th, A, Hc = fit_theta(rank_gaussianize(eta))
        Chat = cov_of(Hc, A, th, d, N)
        wS = np.sort(np.linalg.eigvalsh(Chat))[::-1]
        Ehat = float(wS[R:].sum())
        sig = np.sort(np.abs(np.linalg.eigvalsh(Chat - C_Lam)))[::-1]
        op, kf = float(sig[0]), float(sig[: d - R].sum())
        err = Ehat - E_r
        rows.append(dict(seed=s, Ehat=Ehat, err=err, abs_err=abs(err),
                         op=op, bound_op=(d - R) * op, bound_kf=kf,
                         ratio_op=(d - R) * op / abs(err), ratio_kf=kf / abs(err)))
        print(f"  [{tag}] seed {s}: hatE_r={Ehat:.4f} err={err:+.5f} "
              f"(d-r)op={(d - R) * op:.4f} kf={kf:.4f}", flush=True)
    g = lambda k: np.array([r[k] for r in rows])
    summary = dict(
        Ehat_mean=float(g("Ehat").mean()), Ehat_sd=float(g("Ehat").std(ddof=1)) if seeds > 1 else 0.0,
        abs_err_mean=float(g("abs_err").mean()), abs_err_max=float(g("abs_err").max()),
        bound_op_mean=float(g("bound_op").mean()), bound_kf_mean=float(g("bound_kf").mean()),
        ratio_op_mean=float(g("ratio_op").mean()), ratio_op_min=float(g("ratio_op").min()),
        ratio_kf_mean=float(g("ratio_kf").mean()))
    out = dict(meta=dict(K=K, q=Q, r=R, kappa=KAPPA, nref=nref, meval=meval,
                         N=N, seeds=seeds, seed0=SEED0, d=d),
               oracle=dict(trC=float(np.trace(C_Lam)), E_r=E_r,
                           eigs=[float(x) for x in wL]),
               per_seed=rows, summary=summary)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ehat_tightness_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{tag}] wrote {path}  ratio_op mean={summary['ratio_op_mean']:.1f} "
          f"min={summary['ratio_op_min']:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="all")
    ap.add_argument("--nref", type=int, default=1_000_000)
    ap.add_argument("--meval", type=int, default=200_000)
    ap.add_argument("--N", type=int, default=50_000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "cache"))
    a = ap.parse_args()
    tags = list(SMP) if a.benchmark == "all" else [a.benchmark]
    for tag in tags:
        run_benchmark(tag, a.nref, a.meval, a.N, a.seeds, a.out_dir)


if __name__ == "__main__":
    main()
