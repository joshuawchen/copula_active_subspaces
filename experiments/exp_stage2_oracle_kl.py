"""Composite KL decomposition (Part II): per (N, seed), the marginal, subspace
and reduced-density terms and the measured total KL D_KL(pi_n || pi_hat_n).
Writes the cor44 bound cache and out/exp_stage2_oracle_kl.json.

Usage: python experiments/exp_stage2_oracle_kl.py [--quick]
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

# Example-1 sampler; the others are resolved by _get_sampler(benchmark).
from cas import sample_noise  # noqa: E402
from cas.config import CACHE_DIR, OUT_DIR  # noqa: E402
from cas.noise import log_noise_density  # noqa: E402

# Per-benchmark closed-form log densities for the full noise law (η-space).
def _get_log_density(benchmark: str):
    if benchmark == "banana":
        from cas.noise import log_noise_density
        return log_noise_density
    elif benchmark == "cubic_banana":
        from cas.cubic_banana import log_cubic_banana_density
        return log_cubic_banana_density
    elif benchmark == "ppg":
        from cas.ppg import log_ppg_density
        return log_ppg_density
    elif benchmark == "even_fold":
        from cas.even_fold import log_even_fold_density
        return log_even_fold_density
    elif benchmark == "conformal_cube":
        from cas.conformal_cube import log_conformal_cube_density
        return log_conformal_cube_density
    raise ValueError(f"Unknown benchmark '{benchmark}'")
from cas.densities import ReducedDensityModel  # noqa: E402
from cas.stage1_reference import (  # noqa: E402
    Config, stage_pool, stage_reference,
)
from cas.oracle_kl_marginalization import sample_pi_U  # noqa: E402
from cas.conditional_kl import marginalize_pi_U_with_inner_kl  # noqa: E402





# ---------------------------------------------------------------------
# Benchmark dispatcher
# ---------------------------------------------------------------------

_BENCH_CONFIG = {
    "banana": {
        "d_obs": 20, "r_default": 4,
        "sampler_module": "cas.noise", "sampler_func": "sample_noise",
    },
    "cubic_banana": {
        "d_obs": 25, "r_default": 6,
        "sampler_module": "cas.cubic_banana",
        "sampler_func": "sample_cubic_banana_noise",
    },
    "ppg": {
        "d_obs": 20, "r_default": 3,
        "sampler_module": "cas.ppg", "sampler_func": "sample_ppg_noise",
    },
    "even_fold": {
        "d_obs": 20, "r_default": 4,
        "sampler_module": "cas.even_fold",
        "sampler_func": "sample_even_fold_noise",
    },
    "conformal_cube": {
        "d_obs": 20, "r_default": 4,
        "sampler_module": "cas.conformal_cube",
        "sampler_func": "sample_conformal_cube_noise",
    },
}


def _get_sampler(benchmark: str):
    """Return the noise-sampling callable for the named benchmark."""
    if benchmark not in _BENCH_CONFIG:
        raise ValueError(f"Unknown benchmark '{benchmark}'. "
                          f"Known: {list(_BENCH_CONFIG.keys())}")
    cfg = _BENCH_CONFIG[benchmark]
    import importlib
    mod = importlib.import_module(cfg["sampler_module"])
    return getattr(mod, cfg["sampler_func"])


# =====================================================================
# Per-coord population marginal density
# =====================================================================

def log_standard_logistic_pdf(eta: np.ndarray) -> np.ndarray:
    """log f(eta) for standard logistic density f(eta) = e^{-eta}/(1+e^{-eta})^2.

    Kept for diagnostic comparisons. NOT used as the production reference
    marginal — see _fit_population_marginals for the correct ref-KDE path,
    which handles non-logistic marginals (e.g. banana coords 1, 3).
    """
    # log f(η) = -η - 2 log(1 + e^{-η})
    return -eta - 2.0 * np.log1p(np.exp(-eta))


class _MarginalLogPdf:
    """Lightweight, picklable log-pdf evaluator extracted from a fitted
    ScalarMarginal.

    The Delta_marg workers call only ``log_pdf`` (see the single consumer in
    ``_bound_b_one_cell``). This class copies the 5000-point PchipInterpolator
    and the grid bounds and evaluates the interpolant inside the tabulated
    range; beyond it, it holds the grid-edge value, whereas ScalarMarginal
    evaluates the KDE. At the N_ref = 5e6 reference the KDE has underflowed
    to its 1e-300 floor at the grid edge, so the two agree there. What it
    drops is the fit-time ballast (``x_sorted``, ``_x_train``, the 5M-node CDF
    interpolators, and the KDE itself): the full 20-coordinate object pickles
    to ~9.6 GB at N_ref = 5e6, and re-serializing that with pickle.dumps for
    the ProcessPoolExecutor doubles resident memory; the evaluator list
    pickles to a few MB.
    """

    def __init__(self, m):
        self._logpdf_spline = m._logpdf_spline
        self._logpdf_left = m._logpdf_left
        self._logpdf_right = m._logpdf_right
        self._logpdf_xmin_pad = m._logpdf_xmin_pad
        self._logpdf_xmax_pad = m._logpdf_xmax_pad

    def log_pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x, dtype=float)
        below = x < self._logpdf_xmin_pad
        above = x > self._logpdf_xmax_pad
        mid = ~(below | above)
        out[below] = self._logpdf_left
        out[above] = self._logpdf_right
        out[mid] = self._logpdf_spline(x[mid])
        return out


def _ensure_marginal_evals(
    benchmark: str, N_ref: int, seed: int, cache_dir: str,
) -> list:
    """Return the list of d _MarginalLogPdf evaluators, via a small cache.

    Cache key mirrors _fit_population_marginals with an ``_evals`` suffix.
    Resolution order: (1) load the few-MB evals cache if present; (2) else
    load the big ScalarMarginal pickle if present (one survivable ~10 GB
    resident copy; this reuses an already-landed fit), extract, cache the
    evals, and drop the big objects; (3) else fit from scratch (the fit-time
    transient is the same single ~10 GB peak that completes on a 16 GB
    machine), extract, cache, drop. In every path the caller receives only
    the evaluators, so the parent never holds the big objects at
    pickle.dumps time.
    """
    import pickle

    os.makedirs(cache_dir, exist_ok=True)
    evals_path = os.path.join(
        cache_dir, f"pop_marginal_evals_{benchmark}_N{N_ref}_seed{seed}.pkl"
    )
    if os.path.exists(evals_path):
        with open(evals_path, "rb") as f:
            return pickle.load(f)

    marginals = _fit_population_marginals(
        benchmark=benchmark, N_ref=N_ref, seed=seed, cache_dir=cache_dir,
    )
    evals = [_MarginalLogPdf(m) for m in marginals]
    del marginals

    tmp = evals_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(evals, f)
    os.replace(tmp, evals_path)
    return evals


def _fit_population_marginals(
    benchmark: str, N_ref: int, seed: int, cache_dir: str,
) -> list:
    """Return a list of d ScalarMarginal KDEs fit on N_ref samples from π_n.

    These serve as the *reference* population marginals (treated as truth)
    in the Δ_marg MC integral. Necessary because banana's W2 coords are
    not standard logistic (kurt=40, skew=5).

    Cache key: (benchmark, N_ref, seed). Pickled to disk on first call;
    reused on subsequent calls of the same experiment.
    """
    import pickle
    from cas.reduced_density import ScalarMarginal

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"pop_marginals_{benchmark}_N{N_ref}_seed{seed}.pkl"
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[pop-marginals] fitting {benchmark} N_ref={N_ref} (cached after)",
          flush=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    sampler = _get_sampler(benchmark)
    # Chunked sampling to keep peak memory bounded
    chunk = 500_000
    eta_chunks = []
    remaining = N_ref
    while remaining > 0:
        c = min(chunk, remaining)
        eta_chunks.append(sampler(c, rng))
        remaining -= c
    eta_ref = np.vstack(eta_chunks)

    d = eta_ref.shape[1]
    marginals = [ScalarMarginal(eta_ref[:, i]) for i in range(d)]
    print(f"[pop-marginals] done in {time.time()-t0:.0f}s, "
          f"saving to {cache_path}", flush=True)

    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(marginals, f)
    os.replace(tmp, cache_path)

    return marginals


# =====================================================================
# Bound-B decomposition per cell
# =====================================================================

def _decomp_one_cell(
    N: int, seed: int, V_r_pop_list: list,
    benchmark: str,
    M_kl: int, M_aux: int, n_mode_starts: int,
    seed_offset: int,
    use_secondary_rank: bool,
    c_cov: float, c_sob: float,
    K_inner: int, q_inner_cap: int,
    M_norm: int,
    r: int,
    K_outer: int, q_outer: int,
    pop_marginals_pickle: bytes,
) -> dict:
    """One (N, seed) cell of the Bound-B decomposition.

    Runs Stage-2 fit at V_r_override = V_r_pop (population basis), then
    measures three KL integrals + the full deployment KL via MC.

    Args:
        N, seed: training-set size and seed.
        V_r_pop_list: population basis as nested Python list (d x r).
        benchmark: noise-law key.
        M_kl: MC sample count for the KL integrals.
        M_aux: inner IS sample count for π_U log-density evaluation.
        n_mode_starts: multi-start Newton restart count.
        seed_offset: per-(N, seed) base seed.
        use_secondary_rank: model convention; False = §3.2 paper.
        c_cov, c_sob: Stage-2 ridge tuning constants.
        K_inner, q_inner_cap, M_norm: Stage-2 model knobs.
        r, K_outer, q_outer: Stage-1 model knobs (used by ReducedDensityModel).

    Returns dict with per-cell measurements.
    """
    t0 = time.time()
    V_r_pop = np.array(V_r_pop_list)
    d = V_r_pop.shape[0]

    # --- Step 1: sample training data and fit Stage 2 at V_r_override ---
    from importlib import import_module
    rng_train = np.random.default_rng(seed_offset + seed)
    _sampler_mod_func = _BENCH_CONFIG[benchmark]
    _sampler = getattr(import_module(_sampler_mod_func["sampler_module"]),
                         _sampler_mod_func["sampler_func"])
    eta_train = _sampler(N, rng_train)

    # Production Stage-2 constants (config.RIDGE_DEFAULT): selected for each
    # estimate by held-out validation, with (c_cov, c_sob) the centres of the
    # search grid. exp_apost_endtoend refits this same model and checks that
    # its normalizer matches the one recorded here.
    ridge_scheme = f"heldout:c_cov={c_cov},c_sob={c_sob}"
    model = ReducedDensityModel(
        r=r, K=K_outer, q=q_outer,
        K_inner=K_inner, q_inner=min(r, q_inner_cap),
        use_secondary_rank=use_secondary_rank,
        M_norm=M_norm, seed=seed,
        inner_ridge_scheme=ridge_scheme,
    ).fit(eta_train, V_r_override=V_r_pop)

    # --- Step 2: Measure D_KL(π_n || π̂_n) by MC ---
    # Sample η from the true noise, compute log π_n(η) - log π̂_n(η), average.
    rng_full = np.random.default_rng(seed_offset + seed + 1_000_000)
    eta_full = _sampler(M_kl, rng_full)
    log_pi_n_func = _get_log_density(benchmark)
    log_pi_n = log_pi_n_func(eta_full)
    log_hat_pi_n = model.evaluate_log_density(eta_full)
    full_kl_integrand = log_pi_n - log_hat_pi_n
    full_kl_mean = float(np.mean(full_kl_integrand))
    full_kl_se = float(np.std(full_kl_integrand, ddof=1) / np.sqrt(M_kl))

    # --- Step 3: Δ_marg = sum_i D_KL(π_i^pop || π̂_i) by MC on the same η ---
    # E_{η~π_n}[sum_i (log π_i^ref(η_i) - log π̂_i(η_i))]
    # where π_i^ref is a high-N KDE reference (precomputed once), NOT
    # standard logistic (which is wrong for banana coords 1, 3).
    import pickle
    pop_marginals = pickle.loads(pop_marginals_pickle)
    log_marg_pop = np.zeros(M_kl)
    log_marg_hat = np.zeros(M_kl)
    for i in range(d):
        log_marg_pop += pop_marginals[i].log_pdf(eta_full[:, i])
        log_marg_hat += model.marginals[i].log_pdf(eta_full[:, i])
    marg_kl_integrand = log_marg_pop - log_marg_hat
    delta_marg_mean = float(np.mean(marg_kl_integrand))
    delta_marg_se = float(np.std(marg_kl_integrand, ddof=1) / np.sqrt(M_kl))

    # --- Step 4: Δ_den = D_KL(π_U^V* || π̂_U^V*) by MC ---
    # Both the draws and log π_U are taken in the rank-Gaussianized
    # coordinates, which is the law π̂_U estimates. The earlier path drew from
    # cas.oracle_kl_marginalization and evaluated log π_U by multi-start
    # Laplace importance sampling; that module takes the mixed latent z = L W
    # for Z, which holds only on the coordinates the mixing leaves alone. The
    # inner integral here is deterministic quadrature, so it carries no
    # importance weights and no effective sample size.
    from cas import exact_reference as er
    _spec = er.spec_for(benchmark)
    _gr = er.grids(_spec)
    rng_u = np.random.default_rng(seed_offset + seed + 2_000_000)
    seed_u = int(rng_u.integers(0, 2**31 - 1))
    U_samples = er.sample_U(M_kl, V_r_pop, _spec, _gr, seed=seed_u)
    log_pi_U_vals, _log_c_rU, _qdiag = er.log_pi_U(
        U_samples, V_r_pop, _spec, _gr, nq=7)

    # log π̂_U(u) via the deployed model's U-coordinate evaluator
    log_hat_pi_U_vals = model.evaluate_log_density_U(U_samples)

    den_integrand = log_pi_U_vals - log_hat_pi_U_vals
    delta_den_mean = float(np.mean(den_integrand))
    delta_den_se = float(np.std(den_integrand, ddof=1) / np.sqrt(M_kl))

    return {
        "N": N, "seed": seed,
        # Full deployment KL (the LHS of Prop 4.5)
        "full_kl_mean": full_kl_mean,
        "full_kl_se": full_kl_se,
        # Diagnostics: decomposes full_kl into the two cross-entropy means
        "log_pi_n_test_mean": float(np.mean(log_pi_n)),
        "log_hat_pi_n_test_mean": float(np.mean(log_hat_pi_n)),
        # Reference marginal cross-entropy diagnostic
        "log_pi_marg_ref_test_mean": float(np.mean(log_marg_pop)),
        "log_pi_marg_hat_test_mean": float(np.mean(log_marg_hat)),
        # Δ_marg
        "delta_marg_mean": delta_marg_mean,
        "delta_marg_se": delta_marg_se,
        # Δ_den
        "delta_den_mean": delta_den_mean,
        "delta_den_se": delta_den_se,
        # Diagnostics of the inner quadrature for π_U
        "delta_den_inner_dim": int(_qdiag["inner_dim"]),
        "delta_den_dropped_variance": float(_qdiag["dropped_variance"]),
        # Model diagnostics
        "log_Z_norm": float(model.log_Z_norm),
        "Z_norm_se": float(model.Z_norm_se),
        "Z_norm_clipfrac": float(model.Z_norm_clipfrac),
        "theta_inner_norm_sq": float(np.sum(model.theta_inner ** 2)),
        # Cell-level metadata
        "M_kl": M_kl, "M_aux": M_aux, "n_mode_starts": n_mode_starts,
        "use_secondary_rank": use_secondary_rank,
        "wall_time_s": time.time() - t0,
    }


def stage_bound_b_decomp(
    cells_to_run: list,           # list of (N, seed) tuples
    V_r_pop: np.ndarray,
    benchmark: str,
    cache_path: str,
    *,
    M_kl: int, M_aux: int, n_mode_starts: int,
    n_workers: int,
    seed_offset: int,
    use_secondary_rank: bool,
    c_cov: float, c_sob: float,
    K_inner: int, q_inner_cap: int,
    M_norm: int,
    r: int, K_outer: int, q_outer: int,
    pop_marginals_pickle: bytes,
) -> list:
    """Per-cell Bound-B decomposition with resumable cache.

    Cache_path is a JSON file containing a list of per-cell dicts.
    Atomic writes via .tmp + rename.
    """
    existing = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
    done = {(c["N"], c["seed"]) for c in existing}
    todo = [(N, s) for (N, s) in cells_to_run if (N, s) not in done]

    if not todo:
        print(f"[Bound B] all {len(existing)} cells cached", flush=True)
        return existing

    print(f"[Bound B] {len(todo)} cells (M_kl={M_kl}, M_aux={M_aux}, "
          f"n_starts={n_mode_starts}, n_workers={n_workers})", flush=True)

    V_r_pop_list = V_r_pop.tolist()

    def _save(items):
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, cache_path)

    cell_args = [
        (N, s, V_r_pop_list, benchmark, M_kl, M_aux, n_mode_starts,
         seed_offset, use_secondary_rank, c_cov, c_sob,
         K_inner, q_inner_cap, M_norm, r, K_outer, q_outer,
         pop_marginals_pickle)
        for (N, s) in todo
    ]

    if n_workers <= 1:
        for args in cell_args:
            res = _decomp_one_cell(*args)
            existing.append(res)
            _save(existing)
            print(f"  N={res['N']:>6}, seed={res['seed']:>2}: "
                  f"full_KL={res['full_kl_mean']:.4f}±{res['full_kl_se']:.4f}  "
                  f"Δ_marg={res['delta_marg_mean']:.4f}  "
                  f"Δ_den={res['delta_den_mean']:.4f}  "
                  f"({res['wall_time_s']:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_decomp_one_cell, *a): a for a in cell_args}
            for fut in as_completed(futures):
                res = fut.result()
                existing.append(res)
                _save(existing)
                print(f"  N={res['N']:>6}, seed={res['seed']:>2}: "
                      f"full_KL={res['full_kl_mean']:.4f}±{res['full_kl_se']:.4f}  "
                      f"Δ_marg={res['delta_marg_mean']:.4f}  "
                      f"Δ_den={res['delta_den_mean']:.4f}", flush=True)

    return existing


# =====================================================================
# Summary
# =====================================================================

def _load_disint_cache(disint_path: str) -> dict:
    """Load the Delta_sub reference written by exp_delta_sub_Z.py, keyed by (N, seed)."""
    if not os.path.exists(disint_path):
        return {}
    with open(disint_path) as f:
        obj = json.load(f)
    rows = obj["cells"] if isinstance(obj, dict) else obj
    return {(r["N"], r["seed"]): {"true_stage1_kl_mean":
                                  r["delta_sub_Z_mean"], **r}
            for r in rows}


def build_summary(cfg, ref, decomp_rows: list, disint_lookup: dict,
                  c_cov: float, c_sob: float) -> dict:
    """Per-N summary of the Bound B decomposition.

    Adds Δ_sub from the conditional Monte Carlo cache where available.
    """
    Ns = sorted(set(c["N"] for c in decomp_rows))
    rows = []
    for N in Ns:
        cells = [c for c in decomp_rows if c["N"] == N]
        full_kls = np.array([c["full_kl_mean"] for c in cells])
        d_margs = np.array([c["delta_marg_mean"] for c in cells])
        d_dens = np.array([c["delta_den_mean"] for c in cells])

        # Δ_sub from conditional Monte Carlo cache
        d_subs = []
        for c in cells:
            key = (c["N"], c["seed"])
            if key in disint_lookup:
                d_subs.append(disint_lookup[key]["true_stage1_kl_mean"])
        d_subs = np.array(d_subs)

        row = {
            "N": N,
            "n_cells": len(cells),
            "full_kl_mean": float(full_kls.mean()),
            "full_kl_std": float(full_kls.std()),
            "full_kl_per_seed": [float(x) for x in full_kls],
            "delta_marg_mean": float(d_margs.mean()),
            "delta_marg_std": float(d_margs.std()),
            "delta_marg_per_seed": [float(x) for x in d_margs],
            "delta_den_mean": float(d_dens.mean()),
            "delta_den_std": float(d_dens.std()),
            "delta_den_per_seed": [float(x) for x in d_dens],
            "delta_sub_n_cells": len(d_subs),
        }
        if len(d_subs) > 0:
            row["delta_sub_mean"] = float(d_subs.mean())
            row["delta_sub_std"] = float(d_subs.std())
            row["delta_sub_per_seed"] = [float(x) for x in d_subs]
            # Sum of three Δ terms (the Prop 4.5 bound) where all are available
            if len(d_subs) == len(cells):
                bound_sum = d_margs + d_subs + d_dens
                row["bound_sum_mean"] = float(bound_sum.mean())
                row["bound_sum_std"] = float(bound_sum.std())
                row["bound_sum_per_seed"] = [float(x) for x in bound_sum]
                # Slack: bound - actual
                slack = bound_sum - full_kls
                row["slack_mean"] = float(slack.mean())
                row["slack_std"] = float(slack.std())
                row["slack_min"] = float(slack.min())

        rows.append(row)

    return {
        "config": asdict(cfg),
        "tuning_constants": {"c_cov": c_cov, "c_sob": c_sob},
        "reference": {
            "V_r_population_source": "stage1_reference stage_reference at N_ref",
            "N_ref": ref["N_ref"],
            "trC_Lambda": float(ref["trC"]),
            "eigvals_top": [float(x) for x in ref["eigvals_top"]],
        },
        "per_N": rows,
    }


def print_summary(summary: dict) -> None:
    """Stdout report of the decomposition."""
    rows = summary["per_N"]
    print(f"\n{'N':>7} {'full KL':>14} {'Δ_marg':>12} {'Δ_sub':>14} "
          f"{'Δ_den':>14} {'Σ Δ (bound)':>14}", flush=True)
    print("-" * 90, flush=True)
    for r in rows:
        full = f"{r['full_kl_mean']:>6.3f}±{r['full_kl_std']:.3f}"
        dm = f"{r['delta_marg_mean']:>5.3f}"
        dd = f"{r['delta_den_mean']:>6.3f}±{r['delta_den_std']:.3f}"
        if "delta_sub_mean" in r:
            ds = f"{r['delta_sub_mean']:>5.3f}±{r['delta_sub_std']:.3f}"
        else:
            ds = "    (n/a)"
        if "bound_sum_mean" in r:
            bs = f"{r['bound_sum_mean']:>5.3f}±{r['bound_sum_std']:.3f}"
        else:
            bs = "    (n/a)"
        print(f"{r['N']:>7} {full:>14} {dm:>12} {ds:>14} {dd:>14} {bs:>14}",
              flush=True)

    # The decomposition is exact: full_KL = Σ Δ + R_N + log A, where R_N is the
    # rank-transform remainder and log A the log mass of the composition. The
    # difference below is therefore R_N + log A plus Monte Carlo error; it is
    # not a bound, and its sign may be either.
    print(f"\nfull_KL - Σ Δ (= R_N + log A, up to Monte Carlo error):", flush=True)
    for r in rows:
        if "bound_sum_per_seed" not in r:
            continue
        diff = np.array(r["full_kl_per_seed"]) - np.array(r["bound_sum_per_seed"])
        print(f"  N={r['N']}: mean {diff.mean():+.4f}, "
              f"range [{diff.min():+.4f}, {diff.max():+.4f}]", flush=True)


# =====================================================================
# Driver
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default="banana",
                   choices=["banana", "cubic_banana", "ppg", "even_fold", "conformal_cube"],
                   help="which noise-law benchmark to verify")
    p.add_argument("--quick", action="store_true",
                   help="smoke-test config (~5 min)")
    p.add_argument("--N_pop", type=int, default=None)
    p.add_argument("--N_ref", type=int, default=None)
    p.add_argument("--N_chunk", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=None,
                   help="parallel cells in Stage 1-4 fit")
    p.add_argument("--n_seeds", type=int, default=None)
    p.add_argument("--N_scan", type=str, default=None,
                   help="comma-separated list of N values")
    p.add_argument("--scratch_dir", type=str, default=None)
    p.add_argument("--cache_prefix", type=str, default="cor44",
                   help="shared with exp_stage1_reference; this script "
                        "writes {prefix}_bound_b_*.json and consumes "
                        "{prefix}_disintegration_*.json")
    p.add_argument("--out_dir", type=str, default=None)

    # Stage-2 / regularizer constants
    p.add_argument("--c_cov", type=float, default=3.0)
    p.add_argument("--c_sob", type=float, default=1e-5)
    p.add_argument("--K_inner", type=int, default=5)
    p.add_argument("--q_inner_cap", type=int, default=3)
    p.add_argument("--K_outer", type=int, default=None,
                   help="Stage-1 reference degree K1 (overrides Config default; "
                        "use (4,2) for the coarse-reference panel that shows all "
                        "three bound terms, (4,3) for the deployment-consistent "
                        "panel)")
    p.add_argument("--q_outer", type=int, default=None,
                   help="Stage-1 reference interaction order q1")
    p.add_argument("--M_norm", type=int, default=8192)
    p.add_argument("--use_secondary_rank", action="store_true",
                   help="legacy mode; default is paper §3.2 (False)")

    # MC budget
    p.add_argument("--M_kl", type=int, default=20_000,
                   help="MC sample count for KL integrals")
    p.add_argument("--M_aux", type=int, default=20_000,
                   help="inner IS sample count per u for π_U evaluation. "
                        "Lower than the conditional Monte Carlo default since here we "
                        "use M_kl outer u's and only need rel_SE on log π_U")
    p.add_argument("--n_mode_starts", type=int, default=20)
    p.add_argument("--decomp_n_workers", type=int, default=None,
                   help="parallel workers for the Bound-B stage")
    p.add_argument("--N_ref_marg", type=int, default=5_000_000,
                   help="sample count for the reference marginal KDE fit "
                        "(treated as 'population' for Δ_marg). 5M is plenty "
                        "for banana; lower in --quick mode.")

    args = p.parse_args()

    bench_cfg = _BENCH_CONFIG[args.benchmark]
    cfg = Config(d_obs=bench_cfg["d_obs"], r=bench_cfg["r_default"])
    if args.quick:
        cfg = cfg.quick()
    noise_sampler = _get_sampler(args.benchmark)
    for name in ["N_pop", "N_ref", "N_chunk", "n_workers",
                 "n_seeds", "K_outer", "q_outer"]:
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.N_scan is not None:
        cfg.N_scan = tuple(int(x) for x in args.N_scan.split(","))

    scratch_dir = (args.scratch_dir
                   or os.path.join(CACHE_DIR,
                                    f"_cor44_scratch_{args.benchmark}"))
    out_dir = (args.out_dir
               or os.path.join(OUT_DIR, f"{args.benchmark}_production"))
    cache_prefix = (args.cache_prefix
                    if args.cache_prefix != "cor44"
                    else f"cor44_{args.benchmark}")
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    decomp_workers = args.decomp_n_workers or cfg.n_workers

    print(f"=== Bound B (Prop 4.5) decomposition [{args.benchmark}]: Δ_marg + Δ_sub + Δ_den ===",
          flush=True)
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")
    print(f"  scratch_dir = {scratch_dir}")
    print(f"  out_dir = {out_dir}")
    print(f"  cache_prefix = {cache_prefix}")
    print(f"  c_cov = {args.c_cov}, c_sob = {args.c_sob}")
    print(f"  K_inner = {args.K_inner}, q_inner_cap = {args.q_inner_cap}")
    print(f"  M_norm = {args.M_norm}, "
          f"use_secondary_rank = {args.use_secondary_rank}")
    print(f"  M_kl = {args.M_kl}, M_aux = {args.M_aux}, "
          f"n_mode_starts = {args.n_mode_starts}")
    print(f"  decomp_n_workers = {decomp_workers}")
    print()

    # Reuse Stage 1 (pool) + Stage 3 (reference -> V_r_population) from
    # stage1_reference. Skip Stage 2 (saturation) since we don't need T2 here.
    # Skip Stage 4 (per-N scan) since we drive the fit directly.
    marginals = stage_pool(cfg, noise_sampler, scratch_dir)
    ref_cache = os.path.join(
        CACHE_DIR,
        f"{cache_prefix}_ref_K{cfg.K_outer}_q{cfg.q_outer}_N{cfg.N_ref}.json",
    )
    ref = stage_reference(cfg, noise_sampler, marginals, scratch_dir,
                           ref_cache)
    V_r_pop = np.array(ref["V_r"])
    print(f"[ref] V_r_pop from Stage-3 reference fit "
          f"(N_ref={ref['N_ref']}, |Λ|={ref['dict_size']})", flush=True)
    print(f"      trC_Λ = {ref['trC']:.4f}, "
          f"eigvals_top = {ref['eigvals_top']}", flush=True)

    # Cells to run: cfg.N_scan × cfg.n_seeds, matching exp_stage1_reference
    cells_to_run = [
        (N, s) for N in cfg.N_scan for s in range(cfg.n_seeds)
    ]

    # Bound-B decomposition
    decomp_cache = os.path.join(
        CACHE_DIR,
        f"{cache_prefix}_bound_b_K{cfg.K_outer}_q{cfg.q_outer}_r{cfg.r}.json",
    )
    # Fit population marginals (high-N reference) once; ship only the
    # lightweight log-pdf evaluators to the workers (bit-identical outputs;
    # see _MarginalLogPdf for the memory reason).
    pop_evals = _ensure_marginal_evals(
        benchmark=args.benchmark, N_ref=args.N_ref_marg,
        seed=cfg.seed_pop, cache_dir=scratch_dir,
    )
    import pickle as _pickle
    pop_marginals_pickle = _pickle.dumps(pop_evals)

    decomp_rows = stage_bound_b_decomp(
        cells_to_run, V_r_pop, benchmark=args.benchmark,
        cache_path=decomp_cache,
        M_kl=args.M_kl, M_aux=args.M_aux, n_mode_starts=args.n_mode_starts,
        n_workers=decomp_workers, seed_offset=cfg.seeds_emp_offset,
        use_secondary_rank=args.use_secondary_rank,
        c_cov=args.c_cov, c_sob=args.c_sob,
        K_inner=args.K_inner, q_inner_cap=args.q_inner_cap,
        M_norm=args.M_norm,
        r=cfg.r, K_outer=cfg.K_outer, q_outer=cfg.q_outer,
        pop_marginals_pickle=pop_marginals_pickle,
    )

    # Load the Delta_sub reference (rank-Gaussianized quadrature)
    disint_cache = os.path.join(
        CACHE_DIR, f"delta_sub_Z_cells_{args.benchmark}.json")
    disint_lookup = _load_disint_cache(disint_cache)
    if not disint_lookup:
        print(f"[WARN] no Delta_sub cache at {disint_cache}; "
              f"Delta_sub will be missing. Run "
              f"'python reproduce.py --recompute part2-decomposition' first.",
              flush=True)

    summary = build_summary(cfg, ref, decomp_rows, disint_lookup,
                              args.c_cov, args.c_sob)
    summary_path = os.path.join(out_dir, "exp_stage2_oracle_kl.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}", flush=True)
    print_summary(summary)


if __name__ == "__main__":
    main()
