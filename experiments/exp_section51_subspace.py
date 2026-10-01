"""Subspace-recovery angles against the reference V_r^C for CAS at
(K, q) = (4, 2), PCA and a uniformly random subspace. Writes
cache/exp41_subspace_cache.npz (Example 1) and cache/exp41_subspace_ex23.npz
(Examples 2 and 3).
"""
from __future__ import annotations
import os

# Pin the BLAS thread count before importing numpy, so the summation order,
# and hence the cached angles, reproduce bit for bit.
NW = int(os.environ.get("SEC51_THREADS", "8"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(NW))

import sys
import json
import time

import numpy as np
from scipy.linalg import qr, cho_factor, cho_solve
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from cas import sin_theta_F, sample_noise
from cas.hermite_score_matching import (rank_gaussianize, enumerate_dictionary,
    build_A_b_streaming, assemble_T_from_H, randomized_eig)
from cas.even_fold import sample_even_fold_noise
from cas.conformal_cube import sample_conformal_cube_noise

R = 4
NREF = 1_000_000
SEEDS = int(os.environ.get("SEC51_SEEDS", "8"))
_ns_env = os.environ.get("SEC51_NS", "")
NS = ([int(x) for x in _ns_env.split(",")] if _ns_env
      else [500, 1000, 2000, 5000, 10000, 20000, 50000])
TAGS = ("banana", "even_fold", "conformal_cube")
EX23 = ("even_fold", "conformal_cube")
SMP = {"banana": sample_noise, "even_fold": sample_even_fold_noise,
       "conformal_cube": sample_conformal_cube_noise}

CACHE = os.path.join(REPO, "cache")
OUT_DIR = os.environ.get("SEC51_OUT_DIR", CACHE)
RAW = os.path.join(OUT_DIR, "_exp41_recovery_raw.json")
OUT_BANANA = os.path.join(OUT_DIR, "exp41_subspace_cache.npz")
OUT_EX23 = os.path.join(OUT_DIR, "exp41_subspace_ex23.npz")


def lean_V(eta, K, q, seed=0):
    """Memory-lean rank-R basis; numerically identical V_r to the shared ridge
    solve. Streams one feature block at a time (no Phi retention), adds the
    |alpha|^2 Tikhonov ridge (kappa = 1e-4 * lam_max) on the diagonal in place,
    Cholesky-factorizes in place, and extracts the top-R eigenvectors of T by
    randomized range finding (p=10, q_pwr=2)."""
    Z = rank_gaussianize(eta)
    d = Z.shape[1]
    A = enumerate_dictionary(d, K, q)
    Amat, bvec, _, _, Hpc = build_A_b_streaming(Z, A, K, keep_Phi=False)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=float)
    lam_max = float(eigsh(Amat, k=1, which="LM", return_eigenvectors=False)[0])
    Amat[np.diag_indices_from(Amat)] += (1e-4 * lam_max) * total_deg ** 2
    cf, low = cho_factor(Amat, lower=True, overwrite_a=True)
    theta = cho_solve((cf, low), bvec)
    T = assemble_T_from_H(Hpc, A, theta, d)
    V, _, _ = randomized_eig(T, R, p=10, q_pwr=2, seed=seed)
    return V


def exact_ref(tag):
    """Exact copula active subspace V_r^C for `tag`, via deployment_oracle
    (top-r eigenspace of the exact copula-score covariance, cas.analytic_oracle)."""
    from cas.deployment_oracle import deployment_oracle_subspace
    return deployment_oracle_subspace(tag, R)


def recover():
    """Run or resume the recovery sweep into
    res[tag][method][str(N)] = [sin_theta_F per seed]."""
    res = json.load(open(RAW)) if os.path.exists(RAW) else {}

    def cell(tag, m, N):
        return res.setdefault(tag, {}).setdefault(m, {}).setdefault(str(N), [])

    for tag in TAGS:
        Vref = exact_ref(tag)
        for N in NS:
            c = cell(tag, "cas", N)
            p = cell(tag, "pca", N)
            rn = cell(tag, "random", N)
            for s in range(SEEDS):
                if len(c) > s and len(p) > s and len(rn) > s:
                    continue
                rng = np.random.default_rng(13000 + s)
                eta = SMP[tag](N, rng)
                Vc = lean_V(eta, 4, 2, seed=s)
                Z = rank_gaussianize(eta)
                _, U = np.linalg.eigh(np.cov(Z.T))
                Vp = U[:, ::-1][:, :R]
                Vr, _ = qr(rng.standard_normal((eta.shape[1], R)), mode="economic")
                if len(c) <= s:
                    c.append(float(sin_theta_F(Vc, Vref)))
                if len(p) <= s:
                    p.append(float(sin_theta_F(Vp, Vref)))
                if len(rn) <= s:
                    rn.append(float(sin_theta_F(Vr, Vref)))
            os.makedirs(OUT_DIR, exist_ok=True)
            json.dump(res, open(RAW, "w"))
            print(f"  {tag:15s} N={N:6d}  CAS={np.median(c):.3f}  "
                  f"PCA={np.median(p):.3f}  rnd={np.median(rn):.3f}", flush=True)
    return res


def _save(path, cells, **extra):
    arr = np.empty((), dtype=object)
    arr[()] = cells
    np.savez(path, cells=arr, **extra)
    print(f"[saved] {path}  ({len(cells)} cells)")


def write_caches(res):
    """Transcribe the sweep into the two figure caches (cells-dict schema)."""
    ban = res["banana"]
    cells = {}
    for N_str, vals in ban["cas"].items():
        for s, v in enumerate(vals):
            cells[("cas", s, int(N_str), 4, 2)] = {"angle": float(v)}
    for method in ("pca", "random"):
        for N_str, vals in ban[method].items():
            for s, v in enumerate(vals):
                cells[(method, s, int(N_str))] = {"angle": float(v)}
    n_seeds = max(len(v) for v in ban["cas"].values())
    _save(OUT_BANANA, cells, n_seeds=n_seeds, r_oracle=R)

    cells = {}
    n_seeds = 0
    for ex in EX23:
        block = res[ex]
        for method in ("cas", "pca", "random"):
            for N_str, vals in block[method].items():
                n_seeds = max(n_seeds, len(vals))
                for s, v in enumerate(vals):
                    cells[(ex, method, s, int(N_str))] = {"angle": float(v)}
    _save(OUT_EX23, cells, n_seeds=n_seeds, r_oracle=R, examples=np.array(EX23))


def main():
    t0 = time.time()
    print("=" * 64)
    print(f"Section 5.1 recovery sweep  N={NS}  seeds={SEEDS}  out={OUT_DIR}")
    print("=" * 64, flush=True)
    res = recover()
    write_caches(res)
    print(f"DONE in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
