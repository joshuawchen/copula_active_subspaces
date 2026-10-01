"""A posteriori estimates of the marginal and reduced-density terms of the KL
decomposition (Part II), with weighted Kozachenko-Leonenko entropy estimators
(Berrett, Samworth and Yuan 2019), per (benchmark, N, seed).
Writes cache/apost_e2e_{benchmark}.json.

Usage: python experiments/exp_apost_endtoend.py --benchmark banana
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
CACHE = os.path.join(_REPO, "cache")

_SAMP = {
    "banana": ("cas.noise", "sample_noise"),
    "even_fold": ("cas.even_fold", "sample_even_fold_noise"),
    "conformal_cube": ("cas.conformal_cube", "sample_conformal_cube_noise"),
}


def knn_log_xi(X, ks):
    """Per-point log xi_(j),i (BSY eq. (1)) for each j in ks.

    xi_(j),i = e^{-Psi(j)} V_d (M-1) rho_(j),i^d, so
    log xi_(j),i = d log rho_(j),i + log V_d + log(M-1) - Psi(j).
    Returns dict j -> (M,) array.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    M, d = X.shape
    kmax = max(ks)
    tree = cKDTree(X)
    dist, _ = tree.query(X, k=kmax + 1, workers=-1)   # col 0 = self
    log_Vd = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)
    out = {}
    for j in ks:
        rho = np.maximum(dist[:, j], 1e-300)
        out[j] = d * np.log(rho) + log_Vd + np.log(M - 1.0) - digamma(j)
    return out

# __EXTEND__


def bsy_weights_d4(k):
    """Two-point weights in W^(k) for d = 4: support {floor(k/4), k},
    sum w = 1, w1 g(j1) + w2 g(j2) = 0 with g(j) = Gamma(j + 1/2)/Gamma(j)."""
    j1, j2 = max(1, k // 4), k
    def g(j):
        return float(np.exp(gammaln(j + 0.5) - gammaln(j)))
    w2 = g(j1) / (g(j1) - g(j2))
    return {j1: 1.0 - w2, j2: w2}


def entropy_weighted(X, weights):
    """BSY weighted estimator + their variance estimate V-tilde.

    Hhat_w = (1/M) sum_i sum_j w_j log xi_(j),i
    Vhat_w = max(0, (1/M) sum_i sum_j w_j log^2 xi_(j),i - Hhat_w^2)
    Returns (Hhat_w, SE = sqrt(Vhat_w / M))."""
    logxi = knn_log_xi(X, sorted(weights))
    M = next(iter(logxi.values())).shape[0]
    H = sum(w * float(np.mean(logxi[j])) for j, w in weights.items())
    V = sum(w * float(np.mean(logxi[j] ** 2)) for j, w in weights.items())
    V = max(V - H * H, 0.0)
    return H, float(np.sqrt(V / M))


def entropy_1d(x, k):
    """Unweighted KL estimator in d = 1 (no Gamma constraints in W^(k))."""
    return entropy_weighted(np.asarray(x, float)[:, None], {k: 1.0})


def _refit(bench, N, seed, V_r):
    """Replicate the exp_stage2_oracle_kl cell fit (offset 7000)."""
    import importlib
    from cas.densities import ReducedDensityModel
    mod, fn = _SAMP[bench]
    sampler = getattr(importlib.import_module(mod), fn)
    rng_train = np.random.default_rng(7000 + seed)
    eta_train = sampler(N, rng_train)
    model = ReducedDensityModel(
        r=4, K=4, q=2, K_inner=5, q_inner=3,
        use_secondary_rank=False, M_norm=8192, seed=seed,
        inner_ridge_scheme="heldout:c_cov=3,c_sob=1e-5",
    ).fit(eta_train, V_r_override=V_r)
    return model, sampler


def _load_ref_cells(bench, K=4, q=2, r=4):
    # The reference cells are dictionary-specific: a (K,q) other than the
    # deployed (4,2) needs its OWN cor44 bound_b cache, which is the multi-hour
    # leg of regen-heavy.  Building the path from (K,q) rather than hardcoding
    # 4_q2 means a mismatched run fails to find its cache instead of silently
    # pairing a richer model with the (4,2) reference.
    path = os.path.join(CACHE, f"cor44_{bench}_bound_b_K{K}_q{q}_r{r}.json")
    if not os.path.exists(path):
        return {}
    return {(c["N"], c["seed"]): c for c in json.load(open(path))}


def _one_cell(bench, N, seed, V_r, M, k_den, k_marg, seed_eval, ref_cell):
    t0 = time.time()
    model, sampler = _refit(bench, N, seed, V_r)
    zdiff = (None if ref_cell is None
             else abs(model.log_Z_norm - ref_cell["log_Z_norm"]))
    if zdiff is not None and zdiff > 1e-8:
        # The reference cell must be the same estimate; a different normalizer
        # means a different model (for example a different ridge scheme).
        raise SystemExit(f"{bench} N={N} seed={seed}: refit log Z differs from "
                         f"the reference cell by {zdiff:.3g}; the reference was "
                         f"computed for a different estimate")

    # --- reduced-density term at the reference basis ---
    # The evaluation points are draws of U = V_r^T Z for the rank-Gaussianized
    # Z, which is the law pihat_U estimates. The earlier draw came from
    # cas.oracle_kl_marginalization, which takes the mixed latent z = L W for
    # Z on the assumption that the marginals are exactly logistic; that holds
    # only on the coordinates the mixing leaves alone.
    from cas import exact_reference as er
    _spec = er.spec_for(bench)
    _gr = er.grids(_spec)
    U = er.sample_U(M, V_r, _spec, _gr, seed=seed_eval)
    H_den, se_H_den = entropy_weighted(U, bsy_weights_d4(k_den))
    ce = np.empty(M)
    for i in range(0, M, 50000):
        ce[i:i + 50000] = model.evaluate_log_density_U(U[i:i + 50000])
    ce_mean = float(np.mean(ce))
    se_ce = float(np.std(ce, ddof=1) / np.sqrt(M))
    dhat_den = -H_den - ce_mean
    se_den = se_H_den + se_ce          # Minkowski (conservative)

    # --- marginal term, per coordinate on fresh eta ---
    rng_eval = np.random.default_rng(seed_eval + 500_000)
    eta = sampler(M, rng_eval)
    d = eta.shape[1]
    dhat_marg, se_marg = 0.0, 0.0
    per_coord = []
    for i in range(d):
        H_i, se_H_i = entropy_1d(eta[:, i], k_marg)
        lp = model.marginals[i].log_pdf(eta[:, i])
        ce_i = float(np.mean(lp))
        se_ce_i = float(np.std(lp, ddof=1) / np.sqrt(M))
        dhat_i = -H_i - ce_i
        dhat_marg += dhat_i
        se_marg += (se_H_i + se_ce_i) ** 2
        per_coord.append(dhat_i)
    se_marg = float(np.sqrt(se_marg))  # coordinates independent given fit

    row = {
        "benchmark": bench, "N": N, "seed": seed, "M": M,
        "k_den": k_den, "k_marg": k_marg, "seed_eval": seed_eval,
        "logZ_absdiff": zdiff,
        "H_den": H_den, "ce_den": ce_mean,
        "dhat_den": dhat_den, "se_den": se_den,
        "dhat_marg": dhat_marg, "se_marg": se_marg,
        # log of the mass A of the composition (Part I, eq:cas-estimator):
        # computed with the estimate, it enters the marginal term exactly.
        "log_mass": float(model.log_mass),
        "dhat_marg_per_coord": per_coord,
        "wall_time_s": time.time() - t0,
    }
    if ref_cell is not None:
        row["ref_delta_den"] = ref_cell["delta_den_mean"]
        row["ref_delta_den_se"] = ref_cell["delta_den_se"]
        row["ref_delta_marg"] = ref_cell["delta_marg_mean"]
        row["ref_delta_marg_se"] = ref_cell["delta_marg_se"]
        row["ref_full_kl"] = ref_cell["full_kl_mean"]
        row["ref_full_kl_se"] = ref_cell["full_kl_se"]
    return row


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default="banana",
                   choices=sorted(_SAMP))
    p.add_argument("--N_scan", type=str, default="500,2500,12500,50000,200000")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--M", type=int, default=200_000)
    p.add_argument("--k_den", type=int, default=20,
                   help="weighted-estimator k for the r=4 term "
                        "(support {k//4, k})")
    p.add_argument("--k_marg", type=int, default=10,
                   help="unweighted k for the 1-D marginal terms")
    p.add_argument("--seed_eval_base", type=int, default=777_000)
    p.add_argument("--K", type=int, default=4,
                   help="Stage-1 dictionary degree. A non-default (K,q) reads "
                        "the cor44 ref and bound_b caches AT THAT (K,q) and "
                        "writes apost_e2e_{bench}_K{K}q{q}.json, so a richer "
                        "dictionary is never paired with the (4,2) reference.")
    p.add_argument("--q", type=int, default=2,
                   help="Stage-1 dictionary interaction order.")
    a = p.parse_args()

    bench = a.benchmark
    Ns = [int(x) for x in a.N_scan.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    ref = json.load(open(os.path.join(
        CACHE, f"cor44_{bench}_ref_K{a.K}_q{a.q}_N1000000.json")))
    V_r = np.array(ref["V_r"])
    ref_cells = _load_ref_cells(bench, a.K, a.q)

    suffix = "" if (a.K, a.q) == (4, 2) else f"_K{a.K}q{a.q}"
    cache_path = os.path.join(CACHE, f"apost_e2e_{bench}{suffix}.json")
    rows = json.load(open(cache_path)) if os.path.exists(cache_path) else []
    done = {(r["N"], r["seed"], r["M"], r["k_den"], r["k_marg"])
            for r in rows}

    for N in Ns:
        for s in seeds:
            key = (N, s, a.M, a.k_den, a.k_marg)
            if key in done:
                continue
            row = _one_cell(bench, N, s, V_r, a.M, a.k_den, a.k_marg,
                            a.seed_eval_base + 1000 * Ns.index(N) + s,
                            ref_cells.get((N, s)))
            rows.append(row)
            _atomic_dump(rows, cache_path)
            print(f"[{bench} N={N} seed={s}] "
                  f"Dhat_den={row['dhat_den']:.4f}+-{row['se_den']:.4f} "
                  f"(ref {row.get('ref_delta_den', float('nan')):.4f})  "
                  f"Dhat_marg={row['dhat_marg']:.4f}+-{row['se_marg']:.4f} "
                  f"(ref {row.get('ref_delta_marg', float('nan')):.4f})  "
                  f"logZ|d|={row['logZ_absdiff']}  "
                  f"{row['wall_time_s']:.0f}s", flush=True)
    print(f"cache: {cache_path}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
