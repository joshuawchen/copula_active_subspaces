"""Hierarchical a posteriori estimate of the Hermite truncation term (Part II,
prop:hier-trunc): for nested Stage-1 sets, the oracle and cross-fitted
increments and the ratio beta. Writes cache/hier_trunc_{benchmark}_{enrich}.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cas import sample_noise
from cas.even_fold import sample_even_fold_noise
from cas.conformal_cube import sample_conformal_cube_noise
from cas.analytic_oracle import _spec, _exact_score, compute_C
from cas.hermite_score_matching import (rank_gaussianize, enumerate_dictionary,
                              build_A_b_streaming)

SMP = {"banana": sample_noise, "even_fold": sample_even_fold_noise,
       "conformal_cube": sample_conformal_cube_noise}
K1, Q1, KAPPA, SEED0 = 4, 2, 1e-4, 20260808


def fit_kq(Z, K, Q):
    """Ridge Stage-1 fit at (K, Q); returns (theta, dictionary, raw Gram)."""
    d = Z.shape[1]
    A = enumerate_dictionary(d, K, Q)
    Am, b, _, _, _ = build_A_b_streaming(Z, A, K, keep_Phi=False)
    Araw = Am.copy()
    td = np.array([sum(dg) for (_, dg) in A], float)
    lam = float(eigsh(Am, k=1, which="LM", return_eigenvectors=False)[0])
    Am[np.diag_indices_from(Am)] += (KAPPA * lam) * td ** 2
    return np.linalg.solve(Am, b), A, Araw


def gram_kq(Z, K, Q):
    """Raw gradient Gram on Z at (K, Q), no ridge, no solve."""
    A = enumerate_dictionary(Z.shape[1], K, Q)
    Am, _, _, _, _ = build_A_b_streaming(Z, A, K, keep_Phi=False)
    return Am, A


def pad_into(theta_s, A_s, A_l):
    """Embed small-dictionary coefficients into large-dictionary coords."""
    pos = {tuple(dg): i for i, (_, dg) in enumerate(A_l)}
    out = np.zeros(len(A_l))
    for th, (_, dg) in zip(theta_s, A_s):
        out[pos[tuple(dg)]] = th
    return out


def run(tag, nref, trc_nref, ns, seeds, enrich):
    Kp, Qp = (K1 + 1, Q1) if enrich == "K" else (K1, Q1 + 1)
    _, _, trC = compute_C(_spec(tag), n_ref=trc_nref)
    Zref, _ = _exact_score(_spec(tag), n_ref=nref)
    thL, AL, AmL = fit_kq(Zref, K1, Q1)
    thP, AP, AmP = fit_kq(Zref, Kp, Qp)
    trC_L = float(thL @ AmL @ thL)
    trC_P = float(thP @ AmP @ thP)
    herm2_L = max(trC - trC_L, 0.0)
    herm2_P = max(trC - trC_P, 0.0)
    eta2 = max(trC_P - trC_L, 0.0)
    beta = float(np.sqrt(herm2_P / herm2_L)) if herm2_L > 0 else float("nan")
    out = {"tag": tag, "K1": K1, "q1": Q1, "enrich": enrich,
           "Kprime": Kp, "qprime": Qp, "nref": nref,
           "trC": float(trC), "trC_L": trC_L, "trC_Lprime": trC_P,
           "hermite_L": float(np.sqrt(herm2_L)),
           "hermite_Lprime": float(np.sqrt(herm2_P)),
           "eta_oracle": float(np.sqrt(eta2)), "beta_oracle": beta,
           "rows": []}
    print(f"[{tag}] oracle: hermite_L={out['hermite_L']:.4f} "
          f"hermite_L'={out['hermite_Lprime']:.4f} eta={out['eta_oracle']:.4f} "
          f"beta={beta:.4f}", flush=True)
    smp = SMP[tag]
    for N in ns:
        vx, vn = [], []
        for s in range(seeds):
            rng = np.random.default_rng(SEED0 + 100003 * s + N)
            Z = rank_gaussianize(smp(N, rng))
            nA = N // 4
            tLA, aL, _ = fit_kq(Z[:nA], K1, Q1)
            tPA, aP, _ = fit_kq(Z[:nA], Kp, Qp)
            tLB, _, _ = fit_kq(Z[nA:2 * nA], K1, Q1)
            tPB, _, _ = fit_kq(Z[nA:2 * nA], Kp, Qp)
            Amho, _ = gram_kq(Z[2 * nA:], Kp, Qp)
            dA = tPA - pad_into(tLA, aL, aP)
            dB = tPB - pad_into(tLB, aL, aP)
            vx.append(float(dA @ Amho @ dB))
            vn.append(float(np.sqrt(max(dA @ Amho @ dA, 0.0))))
        vx, vn = np.array(vx), np.array(vn)
        sex = float(vx.std(ddof=1) / np.sqrt(len(vx))) if len(vx) > 1 else 0.0
        sen = float(vn.std(ddof=1) / np.sqrt(len(vn))) if len(vn) > 1 else 0.0
        eta_x = float(np.sqrt(max(vx.mean(), 0.0)))
        out["rows"].append({"N": N,
                            "eta2_cross_mean": float(vx.mean()),
                            "eta2_cross_se": sex,
                            "eta_cross": eta_x,
                            "eta_naive_mean": float(vn.mean()),
                            "eta_naive_se": sen,
                            "eta2_cross": [float(x) for x in vx],
                            "eta_naive": [float(x) for x in vn]})
        print(f"[{tag}] N={N}: eta_cross={eta_x:.4f} "
              f"(eta2 {vx.mean():.4f} se {sex:.4f}), "
              f"eta_naive={vn.mean():.4f} (se {sen:.4f}) "
              f"vs oracle eta={out['eta_oracle']:.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="cache")
    p.add_argument("--nref", type=int, default=200000)
    p.add_argument("--trc-nref", type=int, default=1000000)
    p.add_argument("--ns", type=int, nargs="+",
                   default=[500, 2500, 12500, 50000])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--enrich", choices=["K", "q"], default="q")
    p.add_argument("--tags", nargs="+",
                   default=["banana", "even_fold", "conformal_cube"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for tag in a.tags:
        res = run(tag, a.nref, a.trc_nref, a.ns, a.seeds, a.enrich)
        path = os.path.join(a.out, f"hier_trunc_{tag}_{a.enrich}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, path)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
