#!/usr/bin/env python3
"""Stage-1 a priori terms and a posteriori estimate (Part II; read by
fig_stage1_bounds.py). Writes cache/aposteriori_{benchmark}.json.

Usage: python3 experiments/exp_stage1_aposteriori.py --benchmark banana
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import norm

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
K, Q, R, KAPPA, DELTA, SEED0 = 4, 2, 4, 1e-4, 0.05, 20260701


def fit_theta(Z):
    d = Z.shape[1]
    A = enumerate_dictionary(d, K, Q)
    Am, b, _, _, H = build_A_b_streaming(Z, A, K, keep_Phi=False)
    td = np.array([sum(dg) for (_, dg) in A], float)
    lam = float(eigsh(Am, k=1, which="LM", return_eigenvectors=False)[0])
    Am[np.diag_indices_from(Am)] += (KAPPA * lam) * td ** 2
    return np.linalg.solve(Am, b), A, H


def eigd(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    o = np.argsort(-w)
    return w[o], V[:, o]


def transfer_rank(etaA, etaB):
    NA = etaA.shape[0]
    Z = np.empty_like(etaB)
    for j in range(etaA.shape[1]):
        s = np.sort(etaA[:, j])
        u = np.searchsorted(s, etaB[:, j], side="right") / (NA + 1.0)
        Z[:, j] = norm.ppf(np.clip(u, 1.0 / (NA + 1.0), NA / (NA + 1.0)))
    return Z


def cov_at(Hev, A, th, d, M):
    T = assemble_T_from_H(Hev, A, th, d)
    C = T @ T.T / M
    return 0.5 * (C + C.T)


def run_benchmark(tag, nref, meval, ns, seeds):
    smp = SMP[tag]
    t0 = time.time()
    Zev = rank_gaussianize(smp(meval, np.random.default_rng(SEED0 + 999)))
    d = Zev.shape[1]
    Aref = enumerate_dictionary(d, K, Q)
    _, _, _, _, Hev = build_A_b_streaming(Zev, Aref, K, keep_Phi=False)
    th_ref, _, _ = fit_theta(rank_gaussianize(smp(nref, np.random.default_rng(SEED0))))
    C_Lam = cov_at(Hev, Aref, th_ref, d, meval)
    wL, _ = eigd(C_Lam)
    E_r = float(wL[R:].sum())
    sat = os.path.join(REPO, "cache",
                       f"cor44_{tag}_saturation_production_N1000000.json")
    if os.path.exists(sat):
        herm_sq = float(json.load(open(sat))["basis_truncation_sq_estimate"])
    else:
        # The Hermite multi-index truncation term is an a PRIORI quantity read
        # from the saturation study; a new law has no saturation cache yet.  The
        # a POSTERIORI quantities below (rhs, resB, hatEB, m4hat) do not use it,
        # so leave it at zero and report the a posteriori columns only.
        herm_sq = 0.0
        print(f"[{tag}] no saturation cache; hermite_sq = 0 "
              f"(a posteriori columns are unaffected)", flush=True)
    print(f"[{tag}] d={d} nref={nref} meval={meval} trC_Lam={np.trace(C_Lam):.4f} "
          f"E_r={E_r:.4f} hermite_sq={herm_sq:.4f} ({time.time() - t0:.0f}s)",
          flush=True)
    sr = np.sqrt(d - R)
    rows = []
    for N in ns:
        cell = dict(N=N, N2=N // 2, res3=[], penalty=[], resB=[], hatEB=[],
                    m4hat=[], rhs=[], covered=[])
        for s in range(seeds):
            eta = smp(N, np.random.default_rng(SEED0 + 1009 * s + 7 * N))
            # a priori: deployed full-N fit against the reference C_Lambda
            th, A, Hc = fit_theta(rank_gaussianize(eta))
            T = assemble_T_from_H(Hc, A, th, d)
            Cs = T @ T.T / N
            _, Vs = eigd(Cs)
            Ps = Vs[:, :R] @ Vs[:, :R].T
            res3 = float(np.trace(C_Lam) - np.trace(Ps @ C_Lam))
            cell["res3"].append(res3)
            cell["penalty"].append(res3 - E_r)
            # a posteriori: split-sample estimator of cor:aposteriori
            h = N // 2
            thA, AA, _ = fit_theta(rank_gaussianize(eta[:h]))
            ZB = transfer_rank(eta[:h], eta[h:])
            _, _, _, _, HB = build_A_b_streaming(ZB, AA, K, keep_Phi=False)
            TB = assemble_T_from_H(HB, AA, thA, d)
            CB = TB @ TB.T / (N - h)
            wB, VB = eigd(CB)
            PB = VB[:, :R] @ VB[:, :R].T
            hatEB = float(wB[R:].sum())
            m4 = float((np.sum(TB * TB, 0) ** 2).mean())
            rhs = hatEB + sr * np.sqrt(m4 / ((N - h) * DELTA))
            C2A = cov_at(Hev, AA, thA, d, meval)
            resB = float(np.trace(C2A) - np.trace(PB @ C2A))
            cell["resB"].append(resB)
            cell["hatEB"].append(hatEB)
            cell["m4hat"].append(m4)
            cell["rhs"].append(float(rhs))
            cell["covered"].append(float(rhs >= resB))
        rows.append(cell)
        print(f"  N={N:6d} penalty={np.mean(cell['penalty']):.4f} "
              f"resB={np.mean(cell['resB']):.4f} hatEB={np.mean(cell['hatEB']):.4f} "
              f"rhs={np.mean(cell['rhs']):.4f} "
              f"cov={np.mean(cell['covered']):.2f}", flush=True)
    return dict(meta=dict(K=K, q=Q, r=R, d=d, delta=DELTA, nref=nref,
                          meval=meval, seed0=SEED0, ns=ns, seeds=seeds,
                          hermite_src=os.path.basename(sat)),
                trC_Lam=float(np.trace(C_Lam)), E_r=E_r,
                top_eigs=[float(x) for x in wL[:R + 1]],
                hermite_sq=herm_sq, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="all",
                    choices=["all"] + sorted(SMP))
    ap.add_argument("--nref", type=int, default=1000000)
    ap.add_argument("--meval", type=int, default=200000)
    ap.add_argument("--ns", default="500,2500,12500,50000")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--K", type=int, default=K,
                    help="Stage-1 dictionary degree; default is the deployed 4. "
                         "Any non-default (K,q) writes a SEPARATE cache, "
                         "aposteriori_{tag}_K{K}q{q}.json, so the production "
                         "cache is never overwritten by a diagnostic run.")
    ap.add_argument("--q", type=int, default=Q,
                    help="Stage-1 dictionary interaction order; default 2.")
    args = ap.parse_args()
    globals()["K"], globals()["Q"] = args.K, args.q
    suffix = "" if (args.K, args.q) == (4, 2) else f"_K{args.K}q{args.q}"
    ns = [int(x) for x in args.ns.split(",")]
    tags = sorted(SMP) if args.benchmark == "all" else [args.benchmark]
    os.makedirs(os.path.join(REPO, "cache"), exist_ok=True)
    for tag in tags:
        out = run_benchmark(tag, args.nref, args.meval, ns, args.seeds)
        out["K"], out["q"] = args.K, args.q
        path = os.path.join(REPO, "cache", f"aposteriori_{tag}{suffix}.json")
        json.dump(out, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
