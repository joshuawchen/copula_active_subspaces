"""Subspace-only KL of the Part I supplement (fig:random-subspace) at the CAS,
PCA, reference and uniformly random bases, by nested Monte Carlo.
Writes cache/exp_oracle_kl_random_subspace.npz; resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Tuple

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cas.config import CACHE_DIR, R_ORACLE
from cas import (
    sample_noise, log_noise_density, oracle_subspace,
    sample_cubic_banana_noise, log_cubic_banana_density,
    oracle_subspace_cubic_banana, R_CB_TARGET,
    sample_ppg_noise, log_ppg_density,
    oracle_subspace_ppg, R_PPG_TARGET, D_PPG,
    sample_even_fold_noise, log_even_fold_density,
    oracle_subspace_even_fold, R_EF_TARGET,
    sample_conformal_cube_noise, log_conformal_cube_density,
    oracle_subspace_conformal_cube, R_C3_TARGET,
    rank_gaussianize,
    stage1_oracle_kl,
)
from cas.stage1 import cas_fit_streaming
from cas.consistent_marginal import (fit_consistent_marginals, make_log_c_z,
                                      make_sample_z)


# Lazy, per-process cache of the consistent logistic-base marginal estimators.
# Each fit is disk-cached (deterministic in its seed); the caches are created
# before workers launch, so the spawn re-import only loads, never refits.
_CONSISTENT: dict = {}


def _consistent(tag, sample_eta, log_eta_density):
    if tag not in _CONSISTENT:
        fit = fit_consistent_marginals(sample_eta, tag)
        _CONSISTENT[tag] = (make_log_c_z(fit, log_eta_density),
                            make_sample_z(fit, sample_eta))
    return _CONSISTENT[tag]


# ----------------------------------------------------------------------------
# Benchmark registry. Each entry packages everything stage1_oracle_kl needs
# plus the rank-r and oracle subspace.
# ----------------------------------------------------------------------------
def _banana_log_c_z(z):
    return _consistent("banana", sample_noise, log_noise_density)[0](z)


def _banana_sample_z(N, rng):
    return _consistent("banana", sample_noise, log_noise_density)[1](N, rng)


def _cb_log_c_z(z):
    sig = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    eta = np.log(sig) - np.log1p(-sig)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_cubic_banana_density(eta) - log_f_mar.sum(axis=-1)


def _cb_sample_z(N, rng):
    eta = sample_cubic_banana_noise(N, rng)
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(sig)


def _ppg_log_c_z(z):
    sig = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    eta = np.log(sig) - np.log1p(-sig)
    log_f_mar = -eta - 2.0 * np.log1p(np.exp(-eta))
    return log_ppg_density(eta) - log_f_mar.sum(axis=-1)


def _ppg_sample_z(N, rng):
    eta = sample_ppg_noise(N, rng)
    sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    return stats.norm.ppf(sig)


def _ef_log_c_z(z):
    return _consistent("even_fold", sample_even_fold_noise,
                       log_even_fold_density)[0](z)


def _ef_sample_z(N, rng):
    return _consistent("even_fold", sample_even_fold_noise,
                       log_even_fold_density)[1](N, rng)


def _c3_log_c_z(z):
    return _consistent("conformal_cube", sample_conformal_cube_noise,
                       log_conformal_cube_density)[0](z)


def _c3_sample_z(N, rng):
    return _consistent("conformal_cube", sample_conformal_cube_noise,
                       log_conformal_cube_density)[1](N, rng)


BENCHMARKS = {
    "banana": dict(
        sample_eta=sample_noise,
        log_c_z=_banana_log_c_z,
        sample_z=_banana_sample_z,
        oracle_subspace=oracle_subspace,
        rank=R_ORACLE,  # 4
        d=20,
    ),
    "cubic_banana": dict(
        sample_eta=sample_cubic_banana_noise,
        log_c_z=_cb_log_c_z,
        sample_z=_cb_sample_z,
        oracle_subspace=oracle_subspace_cubic_banana,
        rank=R_CB_TARGET,  # 6
        d=25,
    ),
    "ppg": dict(
        sample_eta=sample_ppg_noise,
        log_c_z=_ppg_log_c_z,
        sample_z=_ppg_sample_z,
        oracle_subspace=oracle_subspace_ppg,
        rank=R_PPG_TARGET,  # 3 (deployment rank — primary directions only)
        d=D_PPG,
    ),
    "even_fold": dict(
        sample_eta=sample_even_fold_noise,
        log_c_z=_ef_log_c_z,
        sample_z=_ef_sample_z,
        oracle_subspace=oracle_subspace_even_fold,
        rank=R_EF_TARGET,  # 4
        d=20,
    ),
    "conformal_cube": dict(
        sample_eta=sample_conformal_cube_noise,
        log_c_z=_c3_log_c_z,
        sample_z=_c3_sample_z,
        oracle_subspace=oracle_subspace_conformal_cube,
        rank=R_C3_TARGET,  # 4
        d=20,
    ),
}


# ----------------------------------------------------------------------------
# Subspace estimators
# ----------------------------------------------------------------------------
def _pca_subspace(eta: np.ndarray, r: int) -> np.ndarray:
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    Sigma = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sigma)
    return V[:, np.argsort(-w)[:r]]


def _cas_subspace(eta: np.ndarray, r: int, seed: int,
                  K: int = 4, q: int = 2) -> np.ndarray:
    """CAS-fit V_r at (K, q) Hermite class with deployment regularizer."""
    out = cas_fit_streaming(eta, K=K, q=q, r=r, seed=seed)
    return out["V_r"]


def _haar_random_subspace(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(A)
    return Q


# ----------------------------------------------------------------------------
# Cache management
# ----------------------------------------------------------------------------
def _load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=True)
    return z["cells"].item()


def _save_cache(path: str, cells: dict, meta: dict):
    # np.savez auto-appends .npz if missing; force the exact basename via
    # writing to the *parent* and then renaming.
    tmp = path + ".tmp"
    if tmp.endswith(".npz"):
        # savez will not append again
        np.savez(tmp, cells=cells, meta=json.dumps(meta))
    else:
        np.savez(tmp, cells=cells, meta=json.dumps(meta))
        # np.savez wrote to tmp + ".npz"
        os.replace(tmp + ".npz", tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Worker for parallel stage1_oracle_kl evaluation
# ----------------------------------------------------------------------------
def _kl_worker(args):
    """Top-level worker for ProcessPoolExecutor.

    Receives:
      (benchmark, V_r, kind_seed, N_outer, M_inner)
    Returns:
      (kind_seed_tag, kl_est)
    Uses BENCHMARKS to fetch log_c_z + sample_z (picklable module-level fns).
    """
    benchmark, V_r, kind_tag, kl_seed, N_outer, M_inner = args
    spec = BENCHMARKS[benchmark]
    res = stage1_oracle_kl(
        V_r, spec["log_c_z"], spec["sample_z"],
        N_outer=N_outer, M_inner=M_inner,
        rng=np.random.default_rng(kl_seed),
    )
    return kind_tag, float(res["kl_est"])


# ----------------------------------------------------------------------------
# Single-cell evaluation
# ----------------------------------------------------------------------------
def evaluate_block(
    benchmark: str,
    N: int,
    seed: int,
    n_random: int,
    N_outer: int,
    M_inner: int,
    n_workers: int = 1,
) -> dict:
    """Evaluate all V_r choices at (benchmark, N, seed) and return a dict
    keyed by 'oracle', 'cas', 'pca', ('random', j).

    If n_workers > 1, the stage1_oracle_kl calls (one per V_r choice) run
    in a ProcessPoolExecutor.  Each call is fully self-contained, so the
    parallelism is embarrassingly parallel; the only overhead is pickling
    V_r (~1 kB) and the kl_est return value.
    """
    spec = BENCHMARKS[benchmark]
    sample_eta = spec["sample_eta"]
    oracle_fn = spec["oracle_subspace"]
    r = spec["rank"]
    d = spec["d"]

    # Sample training data for CAS / PCA
    rng_train = np.random.default_rng(seed)
    eta_train = sample_eta(N, rng_train)

    # Three "principled" V_r choices
    V_oracle = oracle_fn(r)
    V_cas = _cas_subspace(eta_train, r, seed=seed)
    V_pca = _pca_subspace(eta_train, r)

    # Build the full work list of (tag, V_r, kl_seed) tuples
    work = [
        ("oracle", V_oracle, seed + 20_001),
        ("cas",    V_cas,    seed + 20_002),
        ("pca",    V_pca,    seed + 20_003),
    ]
    rng_haar = np.random.default_rng(seed + 30_000)
    for j in range(n_random):
        V_rand = _haar_random_subspace(d, r, rng_haar)
        work.append((("random", j), V_rand, seed + 40_000 + j))

    out = {}
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        args_list = [
            (benchmark, V, tag, kl_seed, N_outer, M_inner)
            for (tag, V, kl_seed) in work
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for tag, kl in pool.map(_kl_worker, args_list):
                out[tag] = kl
    else:
        # Serial path (debugging / n_workers=1)
        for tag, V_r, kl_seed in work:
            _, kl = _kl_worker((benchmark, V_r, tag, kl_seed, N_outer, M_inner))
            out[tag] = kl
    return out


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(
    benchmarks: list[str],
    N_values: list[int],
    n_seeds: int,
    n_random: int,
    N_outer: int,
    M_inner: int,
    out_path: str,
    n_workers: int = 1,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cells = _load_cache(out_path)

    total_blocks = len(benchmarks) * len(N_values) * n_seeds
    block_idx = 0
    t_start = time.time()
    for benchmark in benchmarks:
        for N in N_values:
            for seed in range(n_seeds):
                block_idx += 1
                if (benchmark, N, seed, "cas") in cells and \
                   (benchmark, N, seed, ("random", n_random - 1)) in cells:
                    print(f"[{block_idx}/{total_blocks}] {benchmark} N={N} seed={seed}: cached")
                    continue
                t0 = time.time()
                block = evaluate_block(
                    benchmark, N, seed, n_random, N_outer, M_inner,
                    n_workers=n_workers,
                )
                for k, v in block.items():
                    cells[(benchmark, N, seed, k)] = v
                meta = dict(
                    benchmarks=benchmarks, N_values=N_values, n_seeds=n_seeds,
                    n_random=n_random, N_outer=N_outer, M_inner=M_inner,
                )
                _save_cache(out_path, cells, meta)
                elapsed = time.time() - t_start
                eta_total = elapsed / block_idx * total_blocks
                print(f"[{block_idx}/{total_blocks}] {benchmark} N={N} seed={seed}: "
                      f"done in {time.time()-t0:.1f}s "
                      f"(oracle KL={block['oracle']:.3f}, "
                      f"CAS KL={block['cas']:.3f}, "
                      f"PCA KL={block['pca']:.3f}); "
                      f"ETA {(eta_total-elapsed)/60:.0f}min")
    print(f"\n[done] all {total_blocks} blocks in {(time.time()-t_start)/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str,
                        default="banana,even_fold,conformal_cube")
    parser.add_argument("--N-values", type=str, default="1000,2000,5000")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--N-outer", type=int, default=1500)
    parser.add_argument("--M-inner", type=int, default=4096)
    parser.add_argument("--out", type=str,
                        default=os.path.join(CACHE_DIR,
                                              "exp_oracle_kl_random_subspace.npz"))
    parser.add_argument("--n-workers", type=int, default=1,
                        help="parallel workers per block (each stage1_oracle_kl "
                             "call runs in its own process). Set to the number "
                             "of physical cores; on M4 Max try 8-10.")
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",")
    N_values = [int(x) for x in args.N_values.split(",")]
    print(f"Settings: benchmarks={benchmarks}, N_values={N_values}, "
          f"n_seeds={args.n_seeds}, n_random={args.n_random}, "
          f"N_outer={args.N_outer}, M_inner={args.M_inner}, "
          f"n_workers={args.n_workers}")

    run(benchmarks, N_values, args.n_seeds, args.n_random,
        args.N_outer, args.M_inner, args.out, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
