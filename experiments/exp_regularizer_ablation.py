"""Cross-example regularizer comparison of the Part I supplement
(tab:cross-bench-results): the held-out log-score of the Stage-2 estimate under
nine regularizers, per (N, seed).
Writes cache/v82_regularizer_ablation_{benchmark}.json; resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, qmc

from cas import sample_noise as sample_banana_noise
from cas import log_noise_density as log_banana_density
from cas import ReducedDensityModel
from cas.config import (R_ORACLE as R_BANANA, K_OUTER, Q_OUTER, K_INNER,
                         Q_INNER_CAP, M_NORM, USE_SECONDARY_RANK)
from cas.hermite_score_matching import (build_A_b_streaming, hermite_norm,
                                rank_gaussianize as rank_gauss_pipeline,
                                evaluate_features_psi_only)
from cas.reduced_density import (eval_g_polynomial,
                                    rank_transform_through_marginals)
from cas.cubic_banana import (
    sample_cubic_banana_noise, log_cubic_banana_density, R_CB_TARGET,
)
from cas.ppg import (
    sample_ppg_noise, log_ppg_density, R_PPG_TARGET,
)
from cas.even_fold import (
    sample_even_fold_noise, log_even_fold_density, R_EF_TARGET,
)
from cas.conformal_cube import (
    sample_conformal_cube_noise, log_conformal_cube_density, R_C3_TARGET,
)


# ----------------------------------------------------------------------------
# Benchmark registry. Each benchmark wraps its (sampler, log-density, rank,
# default N values) so the same ablation code applies uniformly.
# ----------------------------------------------------------------------------
@dataclass
class BenchmarkSpec:
    name: str
    sample_fn: Callable[[int, np.random.Generator], np.ndarray]
    log_density_fn: Callable[[np.ndarray], np.ndarray]
    rank: int                       # r⋆: rank for Stage-1 subspace
    default_N_values: list[int]     # typical N range for this benchmark


# A unified N range for all three benchmarks, designed to bracket every
# regime from data-starved (N=100) to comfortably-sampled (N=100000).
# Some N values may turn out to be redundant after the run; trim post-hoc
# for the paper figure based on what the data shows.
_DEFAULT_N: list[int] = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]


BENCHMARKS: dict[str, BenchmarkSpec] = {
    'banana': BenchmarkSpec(
        name='banana',
        sample_fn=sample_banana_noise,
        log_density_fn=log_banana_density,
        rank=R_BANANA,                          # 4
        default_N_values=list(_DEFAULT_N),
    ),
    'cubic_banana': BenchmarkSpec(
        name='cubic_banana',
        sample_fn=sample_cubic_banana_noise,
        log_density_fn=log_cubic_banana_density,
        rank=R_CB_TARGET,                        # d=25, r⋆=6, cubic-banana
        default_N_values=list(_DEFAULT_N),
    ),
    'ppg': BenchmarkSpec(
        name='ppg',
        sample_fn=sample_ppg_noise,
        log_density_fn=log_ppg_density,
        rank=R_PPG_TARGET,                       # d=20, r⋆=3, PPG (primary)
        default_N_values=list(_DEFAULT_N),
    ),
    'even_fold': BenchmarkSpec(
        name='even_fold',
        sample_fn=sample_even_fold_noise,
        log_density_fn=log_even_fold_density,
        rank=R_EF_TARGET,                        # d=20, r⋆=4, even fold (cusp)
        default_N_values=list(_DEFAULT_N),
    ),
    'conformal_cube': BenchmarkSpec(
        name='conformal_cube',
        sample_fn=sample_conformal_cube_noise,
        log_density_fn=log_conformal_cube_density,
        rank=R_C3_TARGET,                        # d=20, r⋆=4, conformal z^3
        default_N_values=list(_DEFAULT_N),
    ),
}


# ----------------------------------------------------------------------------
# Common solver pieces (shared with exp_constrained_conditioning.py)
# ----------------------------------------------------------------------------
@dataclass
class FitInternals:
    m: ReducedDensityModel
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    total_deg: np.ndarray
    eta_test: np.ndarray
    ll_truth_test: float


def fit_internals(eta_train: np.ndarray, eta_test: np.ndarray,
                   seed: int, bench: BenchmarkSpec) -> FitInternals:
    """Fit Stage-1, return (A, b, C, total_deg) for Stage-2 experiments.
    Stage-2 uses R nominal so we can read m's published-default fit."""
    N_train = eta_train.shape[0]
    r_star = bench.rank
    m = ReducedDensityModel(
        r=r_star, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(r_star, Q_INNER_CAP),
        use_secondary_rank=USE_SECONDARY_RANK, M_norm=M_NORM, seed=seed,
        inner_ridge_scheme="theoretical:c_cov=3,c_sob=1e-5",
    ).fit(eta_train)

    Z = rank_gauss_pipeline(eta_train)
    U = Z @ m.V_r
    U_for_fit = (rank_transform_through_marginals(U, m.u_marginals)
                 if m.use_secondary_rank else U)
    basis = m.A_inner
    total_deg = np.array([sum(deg) for (_, deg) in basis], dtype=int)
    A_emp, b_emp, _, _, _ = build_A_b_streaming(U_for_fit, basis, K_INNER,
                                                   keep_Phi=False)

    K_max = int(total_deg.max())
    r = r_star
    n = len(basis)
    H_vals = np.empty((N_train, r, K_max + 1))
    for j in range(r):
        H_vals[:, j, :] = hermite_norm(U_for_fit[:, j], K_max)
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


def solve_constrained(M: np.ndarray, b: np.ndarray, C: np.ndarray) -> np.ndarray:
    cf, low = cho_factor(M, lower=True)
    M_inv_b = cho_solve((cf, low), b)
    M_inv_CT = cho_solve((cf, low), C.T)
    schur = C @ M_inv_CT
    rhs = C @ M_inv_b
    try:
        lam = np.linalg.solve(schur, rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(schur, rhs, rcond=None)[0]
    return M_inv_b - M_inv_CT @ lam


def renormalize_eval(m, theta_new, eta_eval):
    M_pow2 = 1 << int(np.ceil(np.log2(max(m.M_norm, 2))))
    sampler = qmc.Sobol(d=m.r, scramble=True, seed=m.seed)
    U01 = np.clip(sampler.random(M_pow2), 1e-12, 1 - 1e-12)
    Xi = norm.ppf(U01)
    tg_vals = eval_g_polynomial(Xi, theta_new, m.A_inner, m.K_inner)
    tg_clipped = np.minimum(tg_vals, m.tg_cap)
    max_tg = float(np.max(tg_clipped))
    log_Z = max_tg + float(np.log(np.mean(np.exp(tg_clipped - max_tg))))
    th_save = m.theta_inner.copy()
    lZ_save = m.log_Z_norm
    m.theta_inner = theta_new
    m.log_Z_norm = log_Z
    ll = m.evaluate_log_density(eta_eval).mean()
    m.theta_inner = th_save
    m.log_Z_norm = lZ_save
    return ll


# ----------------------------------------------------------------------------
# Cached evaluator: same as renormalize_eval but precomputes the test-side
# Psi (Hermite-basis features), marginal log-pdfs, and the QMC Sobol Xi
# once per (m, eta_eval).  Subsequent calls become a matrix-vector product
# instead of a full evaluate_log_density pipeline (~180x speedup at large
# test-set sizes).  Used in the CV inner loop and in the final
# per-regularizer evaluation in run_one_cell.
# ----------------------------------------------------------------------------
class CachedEvaluator:
    """Precomputes everything in evaluate_log_density that doesn't depend on
    theta or log_Z, so each call is a single Psi @ theta plus a few adds."""

    def __init__(self, m, eta_eval: np.ndarray):
        self.m = m
        eta_eval = np.asarray(eta_eval, dtype=float)
        N = eta_eval.shape[0]

        # Per-coord marginal log-pdfs (sum over primary marginals)
        log_marg = np.zeros(N)
        for i in range(m.d):
            log_marg += m.marginals[i].log_pdf(eta_eval[:, i])

        # Primary rank-transform + project to U_test
        Z_eval = rank_transform_through_marginals(eta_eval, m.marginals)
        U_test = Z_eval @ m.V_r

        if m.use_secondary_rank:
            # Secondary rank transform + Jacobian
            tilde_U = rank_transform_through_marginals(U_test, m.u_marginals)
            log_jac = np.zeros(N)
            for i in range(m.r):
                log_fU = m.u_marginals[i].log_pdf(U_test[:, i])
                log_phi = norm.logpdf(U_test[:, i])
                log_jac += (log_fU - log_phi)
            U_for_eval = tilde_U
        else:
            log_jac = np.zeros(N)
            U_for_eval = U_test

        # Psi at eta_eval (large — N x |Lambda|); the only theta-independent
        # heavy compute.  This is what makes the cache worthwhile.
        self.Psi_eval = evaluate_features_psi_only(U_for_eval, m.A_inner,
                                                    m.K_inner)
        # Cached const = log_marg + log_jac (per-sample)
        self.log_const = log_marg + log_jac

        # Pre-sample QMC Xi for log_Z computation (also reused across calls)
        M_pow2 = 1 << int(np.ceil(np.log2(max(m.M_norm, 2))))
        sampler = qmc.Sobol(d=m.r, scramble=True, seed=m.seed)
        U01 = np.clip(sampler.random(M_pow2), 1e-12, 1 - 1e-12)
        self.Xi = norm.ppf(U01)
        # Psi at Xi (small — M_pow2 x |Lambda|, e.g. 8192 x 116)
        self.Psi_Xi = evaluate_features_psi_only(self.Xi, m.A_inner,
                                                  m.K_inner)
        self.tg_cap = m.tg_cap

    def __call__(self, theta_new: np.ndarray) -> float:
        """Evaluate held-out log-likelihood at theta_new. Returns scalar mean."""
        # log_Z via QMC
        tg_Xi = self.Psi_Xi @ theta_new
        np.minimum(tg_Xi, self.tg_cap, out=tg_Xi)
        max_tg = float(tg_Xi.max())
        log_Z = max_tg + float(np.log(np.mean(np.exp(tg_Xi - max_tg))))

        # tg at eval points
        tg_eval = self.Psi_eval @ theta_new
        np.minimum(tg_eval, self.tg_cap, out=tg_eval)
        return float((tg_eval + self.log_const - log_Z).mean())


# ----------------------------------------------------------------------------
# Regularizer factories.
#
# A factory takes a FitInternals F, a hyperparameter value, and returns the
# n×n matrix R to add to A. The hyperparameter is ε for fixed variants
# and c (which gets divided by N inside the factory) for c/N variants.
# ----------------------------------------------------------------------------
def R_th_factory(F: FitInternals, c: float) -> np.ndarray:
    """R_th = R_cov(c) + R_sob(δ=1e-5). Published structure."""
    n = F.A.shape[0]
    N = F.m._N if hasattr(F.m, '_N') else len(F.m.theta_inner) * 100  # fallback
    # The right N: training size used to build A_emp. Recover from F:
    # actually we built A on N_train samples. We need N_train here.
    # The cleanest source: pass N_train via FitInternals or recover from A_emp.
    # For safety, the per-class trace is computed from A directly without N.
    # But sigma_k^2 = trace(A_kk) / N requires N. We pass it as an attribute.
    # Reading from F.m: m.theta_inner has shape (n,), no help.
    # Use the cached value:
    N = F._N_train
    A_op = float(np.linalg.norm(F.A, ord=2))
    R_cov = np.zeros(n)
    by_deg: dict[int, list[int]] = {}
    for a, td in enumerate(F.total_deg):
        by_deg.setdefault(int(td), []).append(a)
    for k, idx_list in by_deg.items():
        idx = np.array(idx_list)
        sigma_k_sq = max(float(np.trace(F.A[np.ix_(idx, idx)])) / N, 0.0)
        R_cov[idx] = c * sigma_k_sq
    R_sob = 1e-5 * A_op * (F.total_deg ** 2)
    return np.diag(R_cov + R_sob)


def R_identity_fixed(F, eps):
    return eps * np.eye(F.A.shape[0])


def R_identity_cN(F, c):
    return (c / F._N_train) * np.eye(F.A.shape[0])


def R_sobolev1_fixed(F, eps):
    return eps * np.diag(F.total_deg.astype(float))


def R_sobolev1_cN(F, c):
    return (c / F._N_train) * np.diag(F.total_deg.astype(float))


def R_sobolev2_fixed(F, eps):
    return eps * np.diag(F.total_deg.astype(float) ** 2)


def R_sobolev2_cN(F, c):
    return (c / F._N_train) * np.diag(F.total_deg.astype(float) ** 2)


def R_no_reg(F, _ignored):
    return 1e-10 * np.eye(F.A.shape[0])


# Per-regularizer hyperparameter grid for CV.
# Fixed-ε grids cover a wide log range. c/N grids assume c is ~constant
# across N (the whole point of c/N scaling).
GRIDS = {
    'R_th_CV':         np.array([0.3, 1.0, 3.0, 10.0, 30.0]),
    'identity_fixed':  np.logspace(-5, 0, 16),
    'identity_cN':     np.logspace(0, 4, 12),
    'sobolev1_fixed':  np.logspace(-5, 0, 16),
    'sobolev1_cN':     np.logspace(0, 4, 12),
    'sobolev2_fixed':  np.logspace(-6, -1, 16),
    'sobolev2_cN':     np.logspace(-1, 4, 12),
}

FACTORIES = {
    'R_th_nominal':    (R_th_factory, 3.0),   # nominal c=3, no CV
    'R_th_CV':         (R_th_factory, None),  # CV-tuned c
    'identity_fixed':  (R_identity_fixed, None),
    'identity_cN':     (R_identity_cN, None),
    'sobolev1_fixed':  (R_sobolev1_fixed, None),
    'sobolev1_cN':     (R_sobolev1_cN, None),
    'sobolev2_fixed':  (R_sobolev2_fixed, None),
    'sobolev2_cN':     (R_sobolev2_cN, None),
    'no_reg':          (R_no_reg, 1.0),       # ignored arg
}

# Ordering for the table / figure
REG_ORDER = [
    'R_th_nominal', 'R_th_CV',
    'identity_fixed', 'identity_cN',
    'sobolev1_fixed', 'sobolev1_cN',
    'sobolev2_fixed', 'sobolev2_cN',
    'no_reg',
]


# ----------------------------------------------------------------------------
# Per-(N, seed) experiment block
# ----------------------------------------------------------------------------
def cv_select(F_tr: FitInternals, factory, grid: np.ndarray,
               eta_val: np.ndarray, ll_truth_val: float,
               eval_cache: "CachedEvaluator | None" = None):
    """Coarse log-grid CV + one zoom. Returns (best_hyper, best_val_kl).

    If eval_cache is provided, it's used in place of renormalize_eval for
    the held-out evaluation (a single Psi @ theta instead of a full
    pipeline).  Must be built against eta_val for the F_tr's m.
    """
    if eval_cache is None:
        # Fallback: build one ourselves (the call site should pass it in)
        eval_cache = CachedEvaluator(F_tr.m, eta_val)

    best_kl = float('inf'); best_h = None
    for h in grid:
        try:
            R = factory(F_tr, float(h))
            theta = solve_constrained(F_tr.A + R, F_tr.b, F_tr.C)
            ll = eval_cache(theta)
            kl = ll_truth_val - ll
            if kl < best_kl:
                best_kl = kl; best_h = float(h)
        except Exception:
            continue
    if best_h is None:
        return None, float('inf')
    # Zoom: 3 log-spaced points within [best/3, best*3]
    zoom = np.array([best_h / 3.0, best_h, best_h * 3.0])
    for h in zoom:
        try:
            R = factory(F_tr, float(h))
            theta = solve_constrained(F_tr.A + R, F_tr.b, F_tr.C)
            ll = eval_cache(theta)
            kl = ll_truth_val - ll
            if kl < best_kl:
                best_kl = kl; best_h = float(h)
        except Exception:
            continue
    return best_h, best_kl


def run_one_cell(N_train: int, seed: int, bench: BenchmarkSpec,
                  val_frac: float = 0.3) -> dict:
    """Run all regularizers on one (N, seed, benchmark) cell. Returns a dict
    ready to JSON-serialize."""
    rng = np.random.default_rng(seed)
    eta_full = bench.sample_fn(N_train, rng)
    n_val = max(int(val_frac * N_train), 50)
    eta_train = eta_full[:N_train - n_val]
    eta_val = eta_full[N_train - n_val:]
    eta_test = bench.sample_fn(50000, np.random.default_rng(seed + 1000))
    ll_truth_val = bench.log_density_fn(eta_val).mean()
    ll_truth_test = bench.log_density_fn(eta_test).mean()

    # Fit on train-only for CV-picked variants
    F_tr = fit_internals(eta_train, eta_test, seed, bench)
    F_tr._N_train = N_train - n_val

    # Build CV cache once per F_tr (on eta_val), and the final-eval cache once per F (on eta_test)
    eval_cache_val = CachedEvaluator(F_tr.m, eta_val)

    # Find best hyperparameter per CV-tuned regularizer
    picked = {}
    for reg in REG_ORDER:
        factory, default = FACTORIES[reg]
        if default is not None:
            picked[reg] = default
            continue
        grid = GRIDS[reg]
        h, val_kl = cv_select(F_tr, factory, grid, eta_val, ll_truth_val,
                              eval_cache=eval_cache_val)
        picked[reg] = h

    # Refit on full training (train+val), apply each regularizer's picked
    # hyperparameter, evaluate on test
    F = fit_internals(eta_full, eta_test, seed, bench)
    F._N_train = N_train
    eval_cache_test = CachedEvaluator(F.m, eta_test)

    out = {
        'N': int(N_train),
        'seed': int(seed),
        'benchmark': bench.name,
        'kl_truth_test': float(ll_truth_test),  # = -H_eta
        'kl_ref_R_th_nominal': None,
        'regularizers': {},
    }

    for reg in REG_ORDER:
        factory, _ = FACTORIES[reg]
        h = picked[reg]
        if h is None:
            out['regularizers'][reg] = {'kl': None, 'hyperparam': None,
                                          'error': 'no CV winner'}
            continue
        try:
            R = factory(F, float(h))
            theta = solve_constrained(F.A + R, F.b, F.C)
            ll = eval_cache_test(theta)
            kl = ll_truth_test - ll
            out['regularizers'][reg] = {
                'kl': float(kl), 'hyperparam': float(h),
                'theta_norm': float(np.linalg.norm(theta)),
            }
        except Exception as e:
            out['regularizers'][reg] = {
                'kl': None, 'hyperparam': float(h) if h is not None else None,
                'error': f'{type(e).__name__}: {e}'
            }

    # Establish the R_th_nominal reference for ΔKL fields
    ref = out['regularizers'].get('R_th_nominal', {}).get('kl')
    out['kl_ref_R_th_nominal'] = ref
    for reg, rec in out['regularizers'].items():
        if rec.get('kl') is not None and ref is not None:
            rec['dkl_vs_R_th_nominal'] = float(rec['kl'] - ref)
        else:
            rec['dkl_vs_R_th_nominal'] = None
    return out


# ----------------------------------------------------------------------------
# Atomic write + resume
# ----------------------------------------------------------------------------
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
    for c in cache['cells']:
        if c['N'] == N and c['seed'] == seed:
            return True
    return False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='banana',
                         choices=list(BENCHMARKS.keys()),
                         help='Which benchmark noise law to use.')
    parser.add_argument('--N-values', type=str, default=None,
                         help='Comma-separated N values. Default: '
                              'benchmark-specific (see BENCHMARKS).')
    parser.add_argument('--n-seeds', type=int, default=20)
    parser.add_argument('--val-frac', type=float, default=0.3)
    parser.add_argument('--out', type=str, default=None,
                         help='Output cache path. Default: '
                              'cache/v82_regularizer_ablation_<benchmark>.json')
    args = parser.parse_args()

    bench = BENCHMARKS[args.benchmark]
    if args.N_values is None:
        N_values = list(bench.default_N_values)
    else:
        N_values = [int(x) for x in args.N_values.split(',')]
    out_path = (args.out if args.out is not None
                 else f'cache/v82_regularizer_ablation_{bench.name}.json')

    print(f"Benchmark: {bench.name} (rank r⋆ = {bench.rank})", flush=True)
    print(f"N values: {N_values}", flush=True)
    print(f"Output cache: {out_path}\n", flush=True)

    cache = load_cache(out_path)
    cache['config'] = {
        'benchmark': bench.name,
        'rank': bench.rank,
        'N_values': N_values, 'n_seeds': args.n_seeds,
        'val_frac': args.val_frac,
        'regularizers': REG_ORDER,
        'grids': {k: list(v) for k, v in GRIDS.items()},
    }

    # Filter out cells that belong to a *different* benchmark, just in case
    # the path was reused. Belt-and-suspenders: cache filenames already
    # distinguish benchmarks.
    cache['cells'] = [c for c in cache['cells']
                       if c.get('benchmark', bench.name) == bench.name]

    total_cells = len(N_values) * args.n_seeds
    done = sum(1 for N in N_values for s in range(args.n_seeds)
               if already_done(cache, N, s))
    print(f"Plan: {total_cells} cells, {done} already done, "
          f"{total_cells - done} remaining", flush=True)

    for N in N_values:
        for seed in range(args.n_seeds):
            if already_done(cache, N, seed):
                continue
            t0 = time.time()
            try:
                cell = run_one_cell(N, seed, bench, val_frac=args.val_frac)
                cache['cells'].append(cell)
                save_cache(out_path, cache)
                ref = cell['kl_ref_R_th_nominal']
                summary = []
                for reg in REG_ORDER:
                    rec = cell['regularizers'].get(reg, {})
                    kl = rec.get('kl')
                    if kl is None:
                        summary.append(f"{reg}=ERR")
                    else:
                        d = rec.get('dkl_vs_R_th_nominal')
                        d_str = f"{d:+.3f}" if d is not None else "  N/A"
                        summary.append(f"{reg}={kl:.3f}({d_str})")
                print(f"  N={N:>6} seed={seed:>2}: "
                       f"ref={ref:.3f}  " + "  ".join(summary[:3]) + "  ...",
                       flush=True)
                print(f"    full: " + " | ".join(summary), flush=True)
                print(f"    ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  N={N} seed={seed}: FAILED {type(e).__name__}: {e}",
                       flush=True)

    # Final aggregate
    print(f"\n=== Aggregate on {bench.name}: mean ΔKL ± SEM, wins/n_seeds "
           f"vs R_th_nominal ===")
    print(f"  {'regularizer':>18}", end='')
    for N in N_values:
        print(f"  {f'N={N}':>16}", end='')
    print()
    for reg in REG_ORDER:
        print(f"  {reg:>18}", end='')
        for N in N_values:
            deltas = []
            for c in cache['cells']:
                if c['N'] != N:
                    continue
                d = c['regularizers'].get(reg, {}).get('dkl_vs_R_th_nominal')
                if d is not None:
                    deltas.append(d)
            if not deltas:
                print(f"  {'no data':>16}", end='')
                continue
            d = np.array(deltas)
            wins = int((d < 0).sum())
            print(f"  {d.mean():+.3f}±{d.std()/np.sqrt(len(d)):.3f} {wins:>2}/{len(d):<2}".rjust(16), end='')
        print()
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
