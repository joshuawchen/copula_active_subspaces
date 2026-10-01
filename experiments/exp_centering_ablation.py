"""Centering comparison of the Part I supplement (tab:sm2-centering): the
centered and uncentered Stage-2 solves of one regularized system, per (N, seed).
Writes cache/v83_centering_ablation_{benchmark}.json.

Usage: python experiments/exp_centering_ablation.py --benchmark banana
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Reuse the exact deployed Stage-1 fit, R_th regularizer, constrained solve,
# and cached test-KL evaluator from the regularizer ablation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_regularizer_ablation import (  # noqa: E402
    BENCHMARKS,
    fit_internals,
    R_th_factory,
    CachedEvaluator,
    solve_constrained,
)

_DEFAULT_N = [500, 2500, 12500, 50000]


def solve_unconstrained(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    cf, low = cho_factor(M, lower=True)
    return cho_solve((cf, low), b)


def run_one_cell(N_train: int, seed: int, bench) -> dict:
    rng = np.random.default_rng(seed)
    eta_full = bench.sample_fn(N_train, rng)
    eta_test = bench.sample_fn(50000, np.random.default_rng(seed + 1000))
    ll_truth_test = float(bench.log_density_fn(eta_test).mean())

    F = fit_internals(eta_full, eta_test, seed, bench)
    F._N_train = N_train

    R = R_th_factory(F, 3.0)          # R_cov(c_cov=3) + R_sob(c_sob=1e-5)
    M = F.A + R
    ev = CachedEvaluator(F.m, eta_test)

    theta_con = solve_constrained(M, F.b, F.C)   # deployed estimator
    theta_unc = solve_unconstrained(M, F.b)      # same R, constraint removed

    kl_con = ll_truth_test - ev(theta_con)
    kl_unc = ll_truth_test - ev(theta_unc)

    return {
        'N': int(N_train),
        'seed': int(seed),
        'benchmark': bench.name,
        'kl_constrained': float(kl_con),
        'kl_unconstrained': float(kl_unc),
        # negative => constraint helps (lower KL with the constraint on)
        'dkl_constrained_minus_unconstrained': float(kl_con - kl_unc),
        'centering_residual_constrained': float(np.linalg.norm(F.C @ theta_con)),
        'centering_residual_unconstrained': float(np.linalg.norm(F.C @ theta_unc)),
        'theta_norm_constrained': float(np.linalg.norm(theta_con)),
        'theta_norm_unconstrained': float(np.linalg.norm(theta_unc)),
    }


def load_cache(path):
    if not os.path.exists(path):
        return {'config': None, 'cells': []}
    with open(path) as f:
        return json.load(f)


def save_cache(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def already_done(cache, N, seed):
    return any(c['N'] == N and c['seed'] == seed for c in cache['cells'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', type=str, default='banana',
                   choices=list(BENCHMARKS.keys()))
    p.add_argument('--N-values', type=str, default=None,
                   help='Comma-separated N values. Default: 500,2500,12500,50000')
    p.add_argument('--n-seeds', type=int, default=20)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    bench = BENCHMARKS[args.benchmark]
    N_values = ([int(x) for x in args.N_values.split(',')]
                if args.N_values is not None else list(_DEFAULT_N))
    out_path = (args.out if args.out is not None
                else f'cache/v83_centering_ablation_{bench.name}.json')

    print(f"Benchmark: {bench.name} (rank r* = {bench.rank})", flush=True)
    print(f"N values: {N_values}   n_seeds: {args.n_seeds}", flush=True)
    print(f"Output cache: {out_path}\n", flush=True)

    cache = load_cache(out_path)
    cache['config'] = {
        'benchmark': bench.name, 'rank': bench.rank,
        'N_values': N_values, 'n_seeds': args.n_seeds,
        'regularizer': 'R_th_nominal (c_cov=3, c_sob=1e-5)',
        'comparison': 'constrained (deployed) vs unconstrained, same (A+R, b)',
    }
    cache['cells'] = [c for c in cache['cells']
                      if c.get('benchmark', bench.name) == bench.name]

    for N in N_values:
        for seed in range(args.n_seeds):
            if already_done(cache, N, seed):
                continue
            t0 = time.time()
            try:
                cell = run_one_cell(N, seed, bench)
                cache['cells'].append(cell)
                save_cache(out_path, cache)
                d = cell['dkl_constrained_minus_unconstrained']
                print(f"  N={N:>6} seed={seed:>2}: "
                      f"KL con={cell['kl_constrained']:.4f} "
                      f"unc={cell['kl_unconstrained']:.4f}  "
                      f"dKL={d:+.4f}  "
                      f"res(con/unc)={cell['centering_residual_constrained']:.1e}"
                      f"/{cell['centering_residual_unconstrained']:.1e}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  N={N} seed={seed}: FAILED {type(e).__name__}: {e}",
                      flush=True)

    # Aggregate: mean dKL +/- SEM per N (negative => constraint helps)
    print(f"\n=== {bench.name}: mean dKL (constrained - unconstrained) +/- SEM ===")
    for N in N_values:
        ds = np.array([c['dkl_constrained_minus_unconstrained']
                       for c in cache['cells'] if c['N'] == N])
        if len(ds) == 0:
            print(f"  N={N:>6}: no data")
            continue
        wins = int((ds < 0).sum())
        print(f"  N={N:>6}: {ds.mean():+.4f} +/- {ds.std()/np.sqrt(len(ds)):.4f} "
              f"  (constraint lower in {wins}/{len(ds)} seeds)", flush=True)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
