"""Centering comparison across the Stage-2 degree K_2 at fixed N (Part I
supplement, tab:sm2-centering-k2).
Writes cache/v83_k2_sweep_centering_{benchmark}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Reuse the deployed Stage-1 fit pieces, R_th regularizer, constrained solve,
# and cached test-KL evaluator. fit_internals hardcodes K_INNER, so we
# reimplement it with K_inner as a parameter, referencing era's imported names
# to stay byte-for-byte consistent with the deployed pipeline.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_regularizer_ablation as era  # noqa: E402
from exp_regularizer_ablation import (  # noqa: E402
    BENCHMARKS, FitInternals, CachedEvaluator, solve_constrained, R_th_factory,
)

_DEFAULT_K2 = [3, 4, 5, 6]


def fit_internals_K(eta_train, eta_test, seed, bench, K_inner):
    """fit_internals from exp_regularizer_ablation, with K_inner parameterized."""
    N_train = eta_train.shape[0]
    r_star = bench.rank
    m = era.ReducedDensityModel(
        r=r_star, K=era.K_OUTER, q=era.Q_OUTER, K_inner=K_inner,
        q_inner=min(r_star, era.Q_INNER_CAP),
        use_secondary_rank=era.USE_SECONDARY_RANK, M_norm=era.M_NORM, seed=seed,
        inner_ridge_scheme="heldout:c_cov=3,c_sob=1e-5",
    ).fit(eta_train)

    Z = era.rank_gauss_pipeline(eta_train)
    U = Z @ m.V_r
    U_for_fit = (era.rank_transform_through_marginals(U, m.u_marginals)
                 if m.use_secondary_rank else U)
    basis = m.A_inner
    total_deg = np.array([sum(deg) for (_, deg) in basis], dtype=int)
    A_emp, b_emp, _, _, _ = era.build_A_b_streaming(U_for_fit, basis, K_inner,
                                                    keep_Phi=False)

    K_max = int(total_deg.max())
    r = r_star
    n = len(basis)
    H_vals = np.empty((N_train, r, K_max + 1))
    for j in range(r):
        H_vals[:, j, :] = era.hermite_norm(U_for_fit[:, j], K_max)
    C_mat = np.zeros((r, n))
    for a, (supp, deg) in enumerate(basis):
        for j_idx, j in enumerate(supp):
            alpha_j = deg[j_idx]
            if alpha_j == 0:
                continue
            f = math.sqrt(alpha_j)
            vals = np.full(N_train, f)
            for j2_idx, j2 in enumerate(supp):
                d2 = deg[j2_idx] - 1 if j2_idx == j_idx else deg[j2_idx]
                vals = vals * H_vals[:, j2, d2]
            C_mat[j, a] = float(np.mean(vals))

    return FitInternals(
        m=m, A=A_emp, b=b_emp, C=C_mat, total_deg=total_deg,
        eta_test=eta_test,
        ll_truth_test=bench.log_density_fn(eta_test).mean(),
    )


def solve_unconstrained(M, b):
    cf, low = cho_factor(M, lower=True)
    return cho_solve((cf, low), b)


def run_one_cell(N, seed, bench, K_inner):
    rng = np.random.default_rng(seed)
    eta_full = bench.sample_fn(N, rng)
    eta_test = bench.sample_fn(50000, np.random.default_rng(seed + 1000))
    ll_truth_test = float(bench.log_density_fn(eta_test).mean())

    F = fit_internals_K(eta_full, eta_test, seed, bench, K_inner)
    F._N_train = N

    R = R_th_factory(F, 3.0)
    M = F.A + R
    ev = CachedEvaluator(F.m, eta_test)

    theta_con = solve_constrained(M, F.b, F.C)
    theta_unc = solve_unconstrained(M, F.b)

    kl_con = ll_truth_test - ev(theta_con)
    kl_unc = ll_truth_test - ev(theta_unc)

    return {
        'N': int(N),
        'seed': int(seed),
        'benchmark': bench.name,
        'K_inner': int(K_inner),
        'n_basis': int(F.A.shape[0]),
        'kl_constrained': float(kl_con),
        'kl_unconstrained': float(kl_unc),
        'dkl_constrained_minus_unconstrained': float(kl_con - kl_unc),
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


def already_done(cache, K_inner, seed):
    return any(c['K_inner'] == K_inner and c['seed'] == seed
               for c in cache['cells'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', type=str, default='banana',
                   choices=list(BENCHMARKS.keys()))
    p.add_argument('--N', type=int, default=50000,
                   help='Fixed sample size (default 50000, where the cost shows).')
    p.add_argument('--K2-values', type=str, default=None,
                   help='Comma-separated inner degrees. Default 3,4,5,6')
    p.add_argument('--n-seeds', type=int, default=15)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    bench = BENCHMARKS[args.benchmark]
    K2_values = ([int(x) for x in args.K2_values.split(',')]
                 if args.K2_values is not None else list(_DEFAULT_K2))
    out_path = (args.out if args.out is not None
                else f'cache/v83_k2_sweep_centering_{bench.name}.json')

    print(f"Benchmark: {bench.name} (rank r* = {bench.rank})", flush=True)
    print(f"Fixed N: {args.N}   K_2 values: {K2_values}   n_seeds: {args.n_seeds}",
          flush=True)
    print(f"Output cache: {out_path}\n", flush=True)

    cache = load_cache(out_path)
    cache['config'] = {
        'benchmark': bench.name, 'rank': bench.rank, 'N': args.N,
        'K2_values': K2_values, 'n_seeds': args.n_seeds,
        'regularizer': 'R_th_nominal (c_cov=3, c_sob=1e-5)',
        'comparison': 'constrained (deployed) vs unconstrained, vary K_2 at fixed N',
    }
    cache['cells'] = [c for c in cache['cells']
                      if c.get('benchmark', bench.name) == bench.name]

    for K_inner in K2_values:
        for seed in range(args.n_seeds):
            if already_done(cache, K_inner, seed):
                continue
            t0 = time.time()
            try:
                cell = run_one_cell(args.N, seed, bench, K_inner)
                cache['cells'].append(cell)
                save_cache(out_path, cache)
                d = cell['dkl_constrained_minus_unconstrained']
                print(f"  K2={K_inner} seed={seed:>2}: "
                      f"|Lambda_r|={cell['n_basis']:>3}  "
                      f"KL con={cell['kl_constrained']:.4f} "
                      f"unc={cell['kl_unconstrained']:.4f}  "
                      f"dKL={d:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  K2={K_inner} seed={seed}: FAILED {type(e).__name__}: {e}",
                      flush=True)

    # Aggregate: mean dKL +/- SEM per K_2 (the trend is the signal)
    print(f"\n=== {bench.name} at N={args.N}: mean dKL (constrained - unconstrained) "
          f"+/- SEM vs K_2 ===")
    for K_inner in K2_values:
        ds = np.array([c['dkl_constrained_minus_unconstrained']
                       for c in cache['cells'] if c['K_inner'] == K_inner])
        nb = [c['n_basis'] for c in cache['cells'] if c['K_inner'] == K_inner]
        if len(ds) == 0:
            print(f"  K2={K_inner}: no data")
            continue
        print(f"  K2={K_inner} (|Lambda_r|={nb[0] if nb else '?'}): "
              f"{ds.mean():+.4f} +/- {ds.std()/np.sqrt(len(ds)):.4f}", flush=True)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
