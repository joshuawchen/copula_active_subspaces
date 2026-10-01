"""Stage-1 reference KL by conditional Monte Carlo at each estimated basis
(Part II), joined with the scan cells of cas.stage1_reference. Writes the
cor44 conditional Monte Carlo cache.

Usage: python experiments/exp_stage1_reference.py [--quick]
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

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

# Example-1 sampler; the others are resolved by _get_sampler(benchmark).
from cas import sample_noise  # noqa: E402
from cas.config import CACHE_DIR, OUT_DIR  # noqa: E402
from cas.stage1_reference import (  # noqa: E402
    Config, stage_pool, stage_saturation, stage_reference, stage_scan,
)
from cas.conditional_kl import marginalize_pi_U_with_inner_kl  # noqa: E402
from cas.oracle_kl_marginalization import sample_pi_U  # noqa: E402

# A cell is usable only if nearly every outer u produced a valid inner KL.
# Outer u's are dropped when the quadrature reduction does not apply AND the
# IS fall-back fails its acceptance gate; a few such u's bias the cell mean
# slightly, many mean the cell is measuring the wrong thing. Cells below this
# are still written (with their diagnostics, so the failure is inspectable)
# but carry cell_valid = False, and consumers must drop them.
CELL_MIN_VALID_FRAC: float = 0.98





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
# conditional Monte Carlo stage
# =====================================================================

def _disintegration_one_cell(
    N: int, seed: int, V_hat_list: list, benchmark: str,
    N_outer: int, M_aux: int, n_mode_starts: int, seed_outer: int,
    inner_sampler: str = "laplace_is", n_hmc: int = 5000,
    n_warmup: int = 1500, n_leapfrog: int = 40, gmm_K: int = 16,
) -> dict:
    """One (N, seed) cell of the conditional Monte Carlo.

    Args:
        N, seed: cell coords (for accounting only).
        V_hat_list: (d, r) basis as a nested Python list.
        benchmark: noise-law key.
        N_outer: number of outer u samples from pi_U^V_hat.
        M_aux: IS sample count per inner integral.
        n_mode_starts: multi-start Newton restart count.
        seed_outer: RNG seed for outer u sampling and inner IS.

    Returns:
        dict with the per-cell true Stage-1 KL estimate + diagnostics.
    """
    t0 = time.time()
    V_hat = np.array(V_hat_list)

    # Outer MC: sample u's from pi_U^V_hat
    U = sample_pi_U(N_outer, V_hat, benchmark, seed=seed_outer)

    # Inner IS per u: log_pi_U(u) and inner KL D_KL(pi_{W|U=u} || gamma_{d-r})
    res = marginalize_pi_U_with_inner_kl(
        U, V_hat, benchmark,
        M_aux=M_aux, n_mode_starts=n_mode_starts,
        seed=seed_outer + 1,
        progress_every=None,
        inner_sampler=inner_sampler,
        n_hmc=n_hmc, n_warmup=n_warmup, n_leapfrog=n_leapfrog,
        gmm_K=gmm_K,
    )

    inner_kls = res["inner_kl"]
    # NaN marks a u the quadrature could not handle and whose IS fall-back
    # failed its acceptance gate (cas.conditional_kl). Aggregate over the
    # valid u's and record how many were lost: a plain .mean() here would
    # return NaN, and before the gate existed it returned a finite number
    # poisoned by IS blow-ups of order 1e7 nats.
    valid = np.isfinite(inner_kls)
    n_valid = int(valid.sum())
    n_invalid = int(N_outer - n_valid)
    frac_valid = n_valid / N_outer if N_outer else 0.0
    kls_valid = inner_kls[valid]
    mean_kl = float(kls_valid.mean()) if n_valid else float("nan")
    outer_se = (float(kls_valid.std(ddof=1) / np.sqrt(n_valid))
                if n_valid > 1 else float("nan"))
    # Pooled inner IS variance (gives a lower-bound on uncertainty from
    # the inner sampler; outer_se dominates in practice).
    inner_var = float(np.mean(res["inner_kl_se"][valid] ** 2)) if n_valid else float("nan")
    pooled_inner_se = float(np.sqrt(inner_var / n_valid)) if n_valid else float("nan")

    # Sampler-usage diagnostics (always recorded; trivial for laplace_is)
    sampler_used = res.get("inner_sampler_used", np.array([inner_sampler] * N_outer))
    if isinstance(sampler_used, np.ndarray):
        sampler_used_list = [str(x) for x in sampler_used]
    else:
        sampler_used_list = [str(sampler_used)] * N_outer
    n_gmm_used = sum(1 for s in sampler_used_list if s == "gmm")
    n_laplace_fallback = sum(1 for s in sampler_used_list
                              if s.startswith("laplace_fallback"))
    inner_kl_median = float(np.median(kls_valid)) if n_valid else float("nan")
    # Trimmed-mean (drop top/bottom 5%) is reported as a robust
    # alternative to the mean for paper figures.
    if n_valid >= 20:
        sorted_kls = np.sort(kls_valid)
        ntrim = int(0.05 * n_valid)
        true_kl_trimmed = float(sorted_kls[ntrim:n_valid - ntrim].mean())
    else:
        true_kl_trimmed = mean_kl

    extra = {}
    if inner_sampler == "gmm":
        extra["hmc_accept_rate_median"] = float(np.nanmedian(res["hmc_accept_rate"]))
        extra["hmc_accept_rate_min"] = float(np.nanmin(res["hmc_accept_rate"]))
        extra["hmc_step_size_median"] = float(np.nanmedian(res["hmc_step_size"]))
        extra["gmm_K_used_median"] = int(np.median(res["gmm_K_used"]))
        extra["n_gmm_used"] = int(n_gmm_used)
        extra["n_laplace_fallback"] = int(n_laplace_fallback)

    return {
        "N": N, "seed": seed,
        "true_stage1_kl_mean": mean_kl,
        "true_stage1_kl_median": inner_kl_median,
        "true_stage1_kl_trimmed": true_kl_trimmed,
        "true_stage1_kl_outer_se": outer_se,
        "true_stage1_kl_pooled_inner_se": pooled_inner_se,
        "n_outer_valid": n_valid,
        "n_outer_invalid": n_invalid,
        "frac_outer_valid": float(frac_valid),
        "cell_valid": bool(frac_valid >= CELL_MIN_VALID_FRAC and n_valid > 1),
        "inner_kl_min": float(kls_valid.min()) if n_valid else float("nan"),
        "inner_kl_max": float(kls_valid.max()) if n_valid else float("nan"),
        "rel_se_median": float(np.median(res["rel_se"])),
        "rel_se_max": float(res["rel_se"].max()),
        "M_eff_frac_median": float(np.median(res["M_eff_frac"])),
        "M_eff_frac_min": float(res["M_eff_frac"].min()),
        "n_mode_starts_median": float(np.median(res["n_starts_used"]))
        if "n_starts_used" in res else float("nan"),
        "inner_sampler": inner_sampler,
        "N_outer": N_outer, "M_aux": M_aux, "n_mode_starts": n_mode_starts,
        **extra,
        "wall_time_s": time.time() - t0,
    }


def stage_disintegration_kl(
    scan: list,
    benchmark: str,
    cache_path: str,
    N_outer: int,
    M_aux: int,
    n_mode_starts: int,
    n_workers: int,
    seed_outer_base: int = 20260518,
    inner_sampler: str = "laplace_is",
    n_hmc: int = 5000,
    n_warmup: int = 1500,
    n_leapfrog: int = 40,
    gmm_K: int = 16,
) -> list:
    """Stage 5: per-(N, seed) conditional Monte Carlo for the true Stage-1 oracle KL.

    Joins on (N, seed) with the stage1_reference scan cache (which must contain
    V_hat per cell — patched in 691a7cd). Resumable: cells already in
    cache_path are skipped.

    Args:
        scan: list of cell dicts from stage_scan; each must have V_hat.
        benchmark: noise-law key.
        cache_path: per-cell results JSON. Atomic writes via .tmp + rename.
        N_outer: outer MC sample count.
        M_aux: inner IS sample count per u.
        n_mode_starts: Newton multi-start count.
        n_workers: parallel cell workers.
        seed_outer_base: RNG base seed; per-cell seed = base + 13 * idx.

    Returns the merged list of cell-dicts.
    """
    existing = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
    done = {(r["N"], r["seed"]) for r in existing}
    todo = []
    for idx, cell in enumerate(scan):
        key = (cell["N"], cell["seed"])
        if key in done:
            continue
        if "V_hat" not in cell:
            raise RuntimeError(
                f"Cell N={cell['N']}, seed={cell['seed']} has no V_hat in scan "
                f"cache. Re-run verify_cor44_kl_certificate.py with the patch "
                f"from 691a7cd applied (the scan cache must be regenerated)."
            )
        todo.append((idx, cell))

    if not todo:
        print(f"[Stage 5] all {len(existing)} conditional Monte Carlo cells cached",
              flush=True)
        return existing

    print(f"[Stage 5] conditional Monte Carlo for {len(todo)} cells "
          f"(N_outer={N_outer}, M_aux={M_aux}, n_starts={n_mode_starts}, "
          f"n_workers={n_workers})", flush=True)

    def _save(items):
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, cache_path)

    if n_workers <= 1:
        for idx, cell in todo:
            res = _disintegration_one_cell(
                cell["N"], cell["seed"], cell["V_hat"], benchmark,
                N_outer, M_aux, n_mode_starts,
                seed_outer_base + 13 * idx,
                inner_sampler=inner_sampler,
                n_hmc=n_hmc, n_warmup=n_warmup, n_leapfrog=n_leapfrog,
                gmm_K=gmm_K,
            )
            existing.append(res)
            _save(existing)
            print(f"  N={res['N']:>6}, seed={res['seed']:>2}: "
                  f"true_KL={res['true_stage1_kl_mean']:.4f} ± "
                  f"{res['true_stage1_kl_outer_se']:.4f}, "
                  f"rel_SE_med={res['rel_se_median']:.4f}, "
                  f"ESS_med={res['M_eff_frac_median']:.3f} "
                  f"({res['wall_time_s']:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {}
            for idx, cell in todo:
                fut = ex.submit(
                    _disintegration_one_cell,
                    cell["N"], cell["seed"], cell["V_hat"], benchmark,
                    N_outer, M_aux, n_mode_starts,
                    seed_outer_base + 13 * idx,
                    inner_sampler,
                    n_hmc, n_warmup, n_leapfrog, gmm_K,
                )
                futures[fut] = (cell["N"], cell["seed"])
            for fut in as_completed(futures):
                res = fut.result()
                existing.append(res)
                _save(existing)
                print(f"  N={res['N']:>6}, seed={res['seed']:>2}: "
                      f"true_KL={res['true_stage1_kl_mean']:.4f} ± "
                      f"{res['true_stage1_kl_outer_se']:.4f}, "
                      f"rel_SE_med={res['rel_se_median']:.4f}, "
                      f"ESS_med={res['M_eff_frac_median']:.3f}", flush=True)

    return existing


# =====================================================================
# Summary assembly
# =====================================================================

def build_summary(cfg: Config, ref: dict, sat_data: dict, scan: list,
                   disint: list) -> dict:
    """Three-curve summary for Fig I panel B.

    For each N value: mean/std of sin Theta, 3-term RHS, LSI bound,
    and true Stage-1 KL across seeds.
    """
    trC_Lambda = float(ref["trC"])
    eigvals_top = np.array(ref["eigvals_top"])
    E_r = trC_Lambda - float(np.sum(eigvals_top))
    trC_full = sat_data.get("trC_estimate_full", float("nan"))
    basis_trunc_sq = sat_data.get("basis_truncation_sq_estimate", float("nan"))
    basis_trunc = (
        math.sqrt(basis_trunc_sq) if basis_trunc_sq == basis_trunc_sq
        and basis_trunc_sq >= 0 else float("nan")
    )

    # Join conditional Monte Carlo results onto scan by (N, seed)
    disint_lookup = {(r["N"], r["seed"]): r for r in disint}

    Ns = sorted(set(c["N"] for c in scan))
    rows = []
    for N in Ns:
        cells = [c for c in scan if c["N"] == N]
        sts = np.array([c["sin_theta_F"] for c in cells])
        oracs = np.array([c["oracle_KL_bound"] for c in cells])
        term1 = math.sqrt(E_r) if E_r >= 0 else float("nan")
        term2 = basis_trunc
        term3_per = math.sqrt(2 * trC_Lambda) * sts
        rhs_full = 0.5 * (term1 + term2 + term3_per) ** 2

        # True Stage-1 KL per cell (may be missing for some)
        true_kls = []
        true_kl_ses = []
        for c in cells:
            key = (c["N"], c["seed"])
            if key in disint_lookup:
                true_kls.append(disint_lookup[key]["true_stage1_kl_mean"])
                true_kl_ses.append(
                    disint_lookup[key]["true_stage1_kl_outer_se"]
                )
        true_kls = np.array(true_kls)
        true_kl_ses = np.array(true_kl_ses)

        row = {
            "N": N,
            "n_cells": len(cells),
            "sin_theta_mean": float(sts.mean()),
            "sin_theta_std": float(sts.std()),
            "sin_thetas": [float(s) for s in sts],
            # Curve 1: 3-term RHS
            "rhs_3term_mean": float(rhs_full.mean()),
            "rhs_3term_std": float(rhs_full.std()),
            "rhs_3term_per_seed": [float(x) for x in rhs_full],
            # Curve 2: LSI upper bound (from scan)
            "lsi_bound_mean": float(oracs.mean()),
            "lsi_bound_std": float(oracs.std()),
            "lsi_bound_per_seed": [float(o) for o in oracs],
            # Curve 3: True Stage-1 KL via conditional Monte Carlo
            "true_kl_n_cells": len(true_kls),
        }
        if len(true_kls) > 0:
            row["true_kl_mean"] = float(true_kls.mean())
            row["true_kl_std"] = float(true_kls.std())
            row["true_kl_per_seed"] = [float(x) for x in true_kls]
            row["true_kl_outer_ses"] = [float(x) for x in true_kl_ses]
        rows.append(row)

    return {
        "config": asdict(cfg),
        "reference": {
            "trC_Lambda": trC_Lambda,
            "E_r": E_r,
            "top_eigvals": [float(x) for x in eigvals_top],
            "trC_estimate_full": trC_full,
            "basis_truncation_sq": basis_trunc_sq,
            "lambda0_factor_emp": cfg.lambda0_factor_emp,
        },
        "saturation": sat_data.get("rows", []),
        "per_N": rows,
    }


def print_summary(summary: dict) -> None:
    """Stdout report: sinθ, RHS, LSI, true KL per N."""
    rows = summary["per_N"]
    print(f"\n{'N':>7} {'sin Theta':>14} {'RHS (3-term)':>15} "
          f"{'LSI bound':>13} {'True KL':>14}", flush=True)
    print("-" * 75, flush=True)
    for r in rows:
        true_str = (f"{r['true_kl_mean']:>6.3f}±{r['true_kl_std']:.3f}"
                    if "true_kl_mean" in r else "         (n/a)")
        print(f"{r['N']:>7} "
              f"{r['sin_theta_mean']:>8.3f}±{r['sin_theta_std']:.3f} "
              f"{r['rhs_3term_mean']:>9.3f}±{r['rhs_3term_std']:.3f} "
              f"{r['lsi_bound_mean']:>8.3f}±{r['lsi_bound_std']:.3f} "
              f"{true_str}", flush=True)

    print(f"\nBound validity (Cor 4.4 says true KL <= 3-term RHS):", flush=True)
    for r in rows:
        if "true_kl_mean" not in r:
            continue
        true_per = np.array(r["true_kl_per_seed"])
        rhs_per = np.array(r["rhs_3term_per_seed"])
        n_viol = int(np.sum(true_per > rhs_per))
        min_slack = float((rhs_per - true_per).min())
        print(f"  N={r['N']}: bound>=true? "
              f"{n_viol == 0} ({n_viol}/{len(true_per)} viol); "
              f"min slack={min_slack:.4f}", flush=True)

    print(f"\nT3-only diagnostic (NOT a bound, for comparison):", flush=True)
    for r in rows:
        if "true_kl_mean" not in r:
            continue
        print(f"  N={r['N']}: true KL={r['true_kl_mean']:.4f}, "
              f"T3-only={r['lsi_bound_mean']:.4f}, "
              f"ratio T3/true={r['lsi_bound_mean']/r['true_kl_mean']:.3f}",
              flush=True)


# =====================================================================
# Driver
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default="banana",
                   choices=["banana", "cubic_banana", "ppg", "even_fold", "conformal_cube"],
                   help="which noise-law benchmark to verify")
    p.add_argument("--quick", action="store_true",
                   help="smoke-test config (~3 min)")
    p.add_argument("--r", type=int, default=None,
                   help="override the subspace dimension r (defaults to "
                        "benchmark-specific value: banana=4, ppg=3, cubic_banana=6, "
                        "even_fold=4, conformal_cube=4)")
    p.add_argument("--K_outer", type=int, default=None,
                   help="override the deployment-basis K (max total Hermite degree)")
    p.add_argument("--q_outer", type=int, default=None,
                   help="override the deployment-basis q (max per-coordinate degree)")
    p.add_argument("--N_pop", type=int, default=None)
    p.add_argument("--N_ref", type=int, default=None)
    p.add_argument("--N_sat", type=int, default=None)
    p.add_argument("--N_chunk", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=None)
    p.add_argument("--N_scan", type=str, default=None,
                   help="comma-separated list of N values")
    p.add_argument("--scratch_dir", type=str, default=None)
    p.add_argument("--cache_prefix", type=str, default="cor44",
                   help="shared with verify_cor44_kl_certificate; this script "
                        "also produces cor44_disintegration_*.json")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--sat_lambda_mode", type=str, default=None,
                   choices=["production", "fixed"])
    p.add_argument("--sat_bases", type=str, default=None,
                   help="comma-separated K-q pairs, e.g. 4-2,5-2,5-3,6-3")
    p.add_argument("--skip_saturation", action="store_true",
                   help="skip Stage 2; reuse cached saturation if available")
    p.add_argument("--skip_disintegration", action="store_true",
                   help="run only stage1_reference stages 1-4 (no true KL)")

    # conditional Monte Carlo knobs
    p.add_argument("--N_outer", type=int, default=1000,
                   help="outer MC samples for conditional Monte Carlo KL")
    p.add_argument("--M_aux", type=int, default=100_000,
                   help="inner IS samples per u")
    p.add_argument("--n_mode_starts", type=int, default=20,
                   help="multi-start Newton restart count")
    p.add_argument("--disint_n_workers", type=int, default=None,
                   help="parallel workers for conditional Monte Carlo stage; "
                        "defaults to --n_workers")
    p.add_argument("--inner-sampler", dest="inner_sampler",
                   choices=["laplace_is", "gmm", "quad"], default="laplace_is",
                   help="inner sampler: laplace_is (fast, default), "
                        "gmm (HMC+GMM-IS), or quad (deterministic Gauss-Hermite "
                        "quadrature in the active block; exact, no IS variance "
                        "or outliers). quad ignores the gmm/hmc knobs.")
    p.add_argument("--n_hmc", type=int, default=5000,
                   help="HMC samples per u (gmm sampler only)")
    p.add_argument("--n_warmup", type=int, default=1500,
                   help="HMC warmup samples per u (gmm sampler only)")
    p.add_argument("--n_leapfrog", type=int, default=40,
                   help="HMC leapfrog steps per draw (gmm sampler only)")
    p.add_argument("--gmm_K", type=int, default=16,
                   help="GMM component count (gmm sampler only)")
    p.add_argument("--disint_N", type=str, default=None,
                   help="restrict Stage 5 to these comma-separated N values")
    p.add_argument("--disint_seeds", type=str, default=None,
                   help="restrict Stage 5 to these comma-separated seeds")

    args = p.parse_args()

    bench_cfg = _BENCH_CONFIG[args.benchmark]
    cfg = Config(d_obs=bench_cfg["d_obs"], r=bench_cfg["r_default"])
    if args.quick:
        cfg = cfg.quick()
    if args.r is not None:
        cfg.r = args.r
    if args.K_outer is not None:
        cfg.K_outer = args.K_outer
    if args.q_outer is not None:
        cfg.q_outer = args.q_outer
    noise_sampler = _get_sampler(args.benchmark)
    for name in ["N_pop", "N_ref", "N_sat", "N_chunk", "n_workers",
                 "n_seeds", "sat_lambda_mode"]:
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.N_scan is not None:
        cfg.N_scan = tuple(int(x) for x in args.N_scan.split(","))
    if args.sat_bases is not None:
        cfg.sat_bases = tuple(
            tuple(int(x) for x in pq.split("-"))
            for pq in args.sat_bases.split(",")
        )

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

    disint_workers = args.disint_n_workers or cfg.n_workers

    print(f"=== Cor 4.4 full decomposition (3 curves) [{args.benchmark}] ===", flush=True)
    for k, v in asdict(cfg).items():
        print(f"  {k} = {v}")
    print(f"  scratch_dir = {scratch_dir}")
    print(f"  out_dir = {out_dir}")
    print(f"  cache_prefix = {cache_prefix}")
    if not args.skip_disintegration:
        print(f"  N_outer = {args.N_outer}")
        print(f"  M_aux = {args.M_aux}")
        print(f"  n_mode_starts = {args.n_mode_starts}")
        print(f"  disint_n_workers = {disint_workers}")
        print(f"  inner_sampler = {args.inner_sampler}")
        if args.inner_sampler == "gmm":
            print(f"    n_hmc = {args.n_hmc}, n_warmup = {args.n_warmup}, "
                  f"n_leapfrog = {args.n_leapfrog}, gmm_K = {args.gmm_K}")
    print()

    # Stages 1-4 from cas.stage1_reference
    marginals = stage_pool(cfg, noise_sampler, scratch_dir)

    sat_cache = os.path.join(
        CACHE_DIR,
        f"{cache_prefix}_saturation_{cfg.sat_lambda_mode}_N{cfg.N_sat}.json",
    )
    if args.skip_saturation and os.path.exists(sat_cache):
        with open(sat_cache) as f:
            sat_data = json.load(f)
        print(f"[Stage 2] using cached saturation: {sat_cache}", flush=True)
    elif args.skip_saturation:
        print(f"[Stage 2] skipped (no cache at {sat_cache})", flush=True)
        sat_data = {"rows": [], "partial": True}
    else:
        sat_data = stage_saturation(cfg, noise_sampler, marginals,
                                     scratch_dir, sat_cache)

    ref_cache = os.path.join(
        CACHE_DIR,
        f"{cache_prefix}_ref_K{cfg.K_outer}_q{cfg.q_outer}_N{cfg.N_ref}.json",
    )
    ref = stage_reference(cfg, noise_sampler, marginals, scratch_dir,
                           ref_cache)

    scan_cache = os.path.join(
        CACHE_DIR,
        f"{cache_prefix}_scan_K{cfg.K_outer}_q{cfg.q_outer}_r{cfg.r}.json",
    )
    scan = stage_scan(cfg, noise_sampler, ref, marginals, scratch_dir,
                       scan_cache)

    # Stage 5: conditional Monte Carlo for true Stage-1 KL
    disint = []
    if not args.skip_disintegration:
        disint_cache = os.path.join(
            CACHE_DIR,
            f"{cache_prefix}_disintegration_K{cfg.K_outer}_q{cfg.q_outer}"
            f"_r{cfg.r}_{args.inner_sampler}.json",
        )
        scan_sel = scan
        if args.disint_N is not None:
            keep_N = {int(x) for x in args.disint_N.split(",")}
            scan_sel = [c for c in scan_sel if c["N"] in keep_N]
        if args.disint_seeds is not None:
            keep_s = {int(x) for x in args.disint_seeds.split(",")}
            scan_sel = [c for c in scan_sel if c["seed"] in keep_s]
        if len(scan_sel) != len(scan):
            print(f"[Stage 5] restricted to {len(scan_sel)} of {len(scan)} "
                  f"cells by --disint_N/--disint_seeds", flush=True)
        disint = stage_disintegration_kl(
            scan_sel, benchmark=args.benchmark, cache_path=disint_cache,
            N_outer=args.N_outer, M_aux=args.M_aux,
            n_mode_starts=args.n_mode_starts,
            n_workers=disint_workers,
            inner_sampler=args.inner_sampler,
            n_hmc=args.n_hmc, n_warmup=args.n_warmup,
            n_leapfrog=args.n_leapfrog, gmm_K=args.gmm_K,
        )

    # Summary + report
    summary = build_summary(cfg, ref, sat_data, scan, disint)
    summary_path = os.path.join(out_dir, "cor44_full_decomposition.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}", flush=True)
    print_summary(summary)


if __name__ == "__main__":
    main()
