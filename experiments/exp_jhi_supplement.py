"""J_hi and trace bounds at the trace-optimal and J_hi-optimal bases, with the
measured Stage-1 KL at each (Part II supplement, tab:sm-jhi).

Stages: bounds (population moments and both bounds), truekl (conditional Monte Carlo
Monte Carlo), all.
Usage: python3 experiments/exp_jhi_supplement.py --stage all
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from cas.jhi_cas import (
    J_hi, jhi_grassmann_descent, trace_cas_basis, sin_theta_F,
    orthonormal_complement,
)
from cas.oracle_kl_marginalization import _get_active_callables, sample_pi_U
from cas.conditional_kl import marginalize_pi_U_with_inner_kl


# ---------------------------------------------------------------------------
# Example registry — paper-facing label -> internal benchmark, rank, N_pop
# ---------------------------------------------------------------------------

EXAMPLES = {
    "example1": {"benchmark": "banana",       "r": 4, "N_pop": 2_000_000,
                 "inner_sampler": "gmm",        "N_outer": 50,  "M_aux": 50_000},
    "example2": {"benchmark": "even_fold",    "r": 4, "N_pop": 2_000_000,
                 "inner_sampler": "laplace_is", "N_outer": 200, "M_aux": 50_000},
    "example3": {"benchmark": "conformal_cube", "r": 4, "N_pop": 2_000_000,
                 "inner_sampler": "laplace_is", "N_outer": 200, "M_aux": 50_000},
}

SEED_MOMENTS = 0           # population moments (M, H, C); deterministic
SEED_OUTER_TRACE = 20260521  # outer sample seed for true-KL at V_trace
SEED_OUTER_HI = 20260522     # outer sample seed for true-KL at V_hi

PARTIAL_PATH = os.path.join(_REPO, "out/supplement_table_partial.json")
FINAL_PATH = os.path.join(_REPO, "out/supplement_table_data.json")

# Tracked cache for the Part II supplement table tab:sm-jhi.  `bounds` refreshes
# only the bound columns here (preserving any committed truekl columns); the
# slow `truekl` stage overwrites the full record.
CACHE_PATH = os.path.join(_REPO, "cache/jhi_supplement.json")
_BOUNDS_KEYS = (
    "benchmark_internal_name", "d", "r", "N_pop", "seed",
    "J_trace_at_Vtrace", "J_hi_at_Vtrace", "J_trace_at_Vjhi", "J_hi_at_Vjhi",
    "sin_theta", "H_perp_eigs_min", "H_perp_eigs_max", "H_perp_eigs",
    "descent_iters",
)


def _mirror_bounds_to_cache(results: dict) -> None:
    """Write the bound columns into the committed cache, preserving truekl."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    for label, r_ in results.items():
        entry = cache.get(label, {})
        for k in _BOUNDS_KEYS:
            if k in r_:
                entry[k] = r_[k]
        cache[label] = entry
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  mirrored bounds -> {os.path.relpath(CACHE_PATH, _REPO)}", flush=True)


def _mirror_full_to_cache(results: dict) -> None:
    """Overwrite the committed cache with the full (bounds + truekl) record."""
    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  mirrored full record -> {os.path.relpath(CACHE_PATH, _REPO)}", flush=True)


# ---------------------------------------------------------------------------
# Population moments in W-space (closed-form scores; no KDE, no FD)
# ---------------------------------------------------------------------------

def _sample_W(N: int, benchmark: str, rng: np.random.Generator) -> np.ndarray:
    """Draw N W-space samples for the given benchmark."""
    if benchmark == "banana":
        from cas.noise import sample_banana_W
        return sample_banana_W(N, rng)
    if benchmark == "cubic_banana":
        from cas.cubic_banana import D_CB, N_PAIRS, ALPHA_CB
        W = rng.standard_normal((N, D_CB))
        for k in range(N_PAIRS):
            W_pre = W[:, 2 * k + 1]
            W[:, 2 * k + 1] = (
                (ALPHA_CB / np.sqrt(15.0)) * W[:, 2 * k] ** 3
                + np.sqrt(1.0 - ALPHA_CB ** 2) * W_pre
            )
        return W
    if benchmark == "ppg":
        from cas.ppg import _sample_W as _sample_ppg_W
        return _sample_ppg_W(N, rng)
    if benchmark == "even_fold":
        from cas.even_fold import sample_even_fold_W
        return sample_even_fold_W(N, rng)
    if benchmark == "conformal_cube":
        from cas.conformal_cube import sample_conformal_cube_W
        return sample_conformal_cube_W(N, rng)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def compute_population_moments(benchmark: str, N: int, seed: int) -> dict:
    """Exact (M, H, C) in the rank-Gaussianized coordinates.

    M = E[Z Z^T], H = E[grad log pi_Z grad log pi_Z^T], C = E[S S^T] for the
    copula score S, all from cas.analytic_oracle, which builds the exact
    marginals of the mixed latent z = L W as Gaussian mixtures and applies the
    rank transform Z_k = Phi^{-1}(F_k(z_k)) to them.

    This replaces an earlier surrogate that set Z = W L^T and took the
    coordinates of the mixed latent for the rank-Gaussianized ones. Under the
    orthogonal mixing of the current construction that surrogate is degenerate:
    with L orthogonal, L^{-T} = L, so the score-plus-coordinate combination
    S = grad log pi_Z + Z equals L (grad log pi_W + W), whose padding block
    cancels exactly. C then has exact rank k_act, the trace bound at its own
    top-r eigenspace is zero to machine precision, the complement block of H is
    the identity, and J_hi collapses to a constant of d - r shared by all three
    examples. The exact rank transform is not degenerate, because the mixed marginals are
    not exactly Gaussian and the componentwise transform Phi^{-1} o F_k is not
    the identity.
    """
    from cas.analytic_oracle import _spec, compute_MHC
    return compute_MHC(_spec(benchmark), n_ref=N, seed=seed)


def trace_bound(V_r: np.ndarray, C: np.ndarray) -> float:
    """Thm-4.1 / Cor-4.4 Gaussian-reference trace bound: 0.5 tr((I - V V^T) C)."""
    d = C.shape[0]
    return 0.5 * float(np.trace((np.eye(d) - V_r @ V_r.T) @ C))


# ---------------------------------------------------------------------------
# Stage: bounds
# ---------------------------------------------------------------------------

def run_bounds(only: list | None = None) -> dict:
    os.makedirs(os.path.join(_REPO, "out"), exist_ok=True)
    if os.path.exists(PARTIAL_PATH):
        with open(PARTIAL_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    labels = only or list(EXAMPLES)
    for label in labels:
        cfg = EXAMPLES[label]
        name, r, N = cfg["benchmark"], cfg["r"], cfg["N_pop"]
        print(f"\n=== {label} = {name} (r={r}, N_pop={N:_}) ===", flush=True)

        t0 = time.time()
        mom = compute_population_moments(name, N, SEED_MOMENTS)
        M, H, C, d = mom["M"], mom["H"], mom["C"], mom["d"]
        print(f"  population moments: {time.time() - t0:.1f}s")

        # V_trace = top-r eigvecs of C (closed form). V_hi via Grassmann descent.
        V_trace = trace_cas_basis(C, r)
        t0 = time.time()
        V_hi, info = jhi_grassmann_descent(
            M, H, r, V_init=V_trace, n_iter=5000, tol=1e-10)
        print(f"  Grassmann descent: {time.time() - t0:.2f}s, "
              f"{info['n_iters']} iters, grad_norm {info['final_grad_norm']:.2e}, "
              f"terminated={info['terminated_for']}")

        jt_at_tr = trace_bound(V_trace, C)
        jh_at_tr = J_hi(V_trace, M, H)
        jt_at_hi = trace_bound(V_hi, C)
        jh_at_hi = J_hi(V_hi, M, H)
        st = sin_theta_F(V_trace, V_hi)

        V_perp = orthonormal_complement(V_trace)
        H_perp_eigs = np.linalg.eigvalsh(V_perp.T @ H @ V_perp)

        print(f"  sin theta(V_trace, V_hi) = {st:.4f}")
        print(f"  trace @ V_trace = {jt_at_tr:.4f}, @ V_hi = {jt_at_hi:.4f}")
        print(f"  J_hi  @ V_trace = {jh_at_tr:.4f}, @ V_hi = {jh_at_hi:.4f}")
        print(f"  H_perp eigvals at V_trace: "
              f"[{H_perp_eigs.min():.3f}, {H_perp_eigs.max():.3f}]")

        results[label] = {
            "benchmark_internal_name": name,
            "d": d, "r": r, "N_pop": N, "seed": SEED_MOMENTS,
            "J_trace_at_Vtrace": jt_at_tr,
            "J_hi_at_Vtrace": jh_at_tr,
            "J_trace_at_Vjhi": jt_at_hi,
            "J_hi_at_Vjhi": jh_at_hi,
            "sin_theta": st,
            "H_perp_eigs_min": float(H_perp_eigs.min()),
            "H_perp_eigs_max": float(H_perp_eigs.max()),
            "H_perp_eigs": [float(x) for x in H_perp_eigs],
            "descent_iters": info["n_iters"],
            # 'complete' flips to True once true-KL columns are added in 'truekl'
            "complete": results.get(label, {}).get("complete", False),
        }
        np.savez(os.path.join(_REPO, f"out/V_for_supplement_{label}.npz"),
                 V_trace=V_trace, V_jhi=V_hi, M=M, H=H, C=C)
        with open(PARTIAL_PATH, "w") as f:
            json.dump(results, f, indent=2)

    _mirror_bounds_to_cache(results)
    _print_bounds_table(results)
    return results


def _print_bounds_table(results: dict) -> None:
    print(f"\n{'Example':<12} {'sin th':>8} "
          f"{'J_tr@Vtr':>10} {'J_hi@Vtr':>10} "
          f"{'J_tr@Vjh':>10} {'J_hi@Vjh':>10}")
    for label in ["example1", "example2", "example3"]:
        if label not in results:
            continue
        r_ = results[label]
        print(f"{label:<12} {r_['sin_theta']:>8.4f} "
              f"{r_['J_trace_at_Vtrace']:>10.4f} {r_['J_hi_at_Vtrace']:>10.4f} "
              f"{r_['J_trace_at_Vjhi']:>10.4f} {r_['J_hi_at_Vjhi']:>10.4f}")


# ---------------------------------------------------------------------------
# Stage: true KL via conditional Monte Carlo
# ---------------------------------------------------------------------------

def _summarize_kl(kl_arr: np.ndarray, M_aux: int, extra: dict | None = None) -> dict:
    n = len(kl_arr)
    s = np.sort(kl_arr)
    nt = int(0.05 * n)
    trimmed = float(s[nt:n - nt].mean()) if n - 2 * nt > 0 else float(s.mean())
    out = {
        "true_kl_mean": float(kl_arr.mean()),
        "true_kl_median": float(np.median(kl_arr)),
        "true_kl_trimmed": trimmed,
        "outer_se": float(kl_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "N_outer": n, "M_aux": M_aux,
    }
    if extra:
        out.update(extra)
    return out


def _true_kl_laplace(V, benchmark, N_outer, M_aux, seed_outer) -> dict:
    """conditional Monte Carlo MC at V with the Laplace-IS inner sampler (examples 2, 3)."""
    U = sample_pi_U(N_outer, V, benchmark, seed=seed_outer)
    t0 = time.time()
    res = marginalize_pi_U_with_inner_kl(
        U, V, benchmark, M_aux=M_aux, n_mode_starts=20,
        seed=seed_outer + 1, progress_every=None, inner_sampler="laplace_is",
    )
    kl_arr = np.asarray(res["inner_kl"], dtype=float)
    return _summarize_kl(kl_arr, M_aux, {
        "rel_se_median": float(np.median(res["rel_se"])),
        "M_eff_frac_median": float(np.median(res["M_eff_frac"])),
        "wall_time_s": time.time() - t0,
    })


def _true_kl_gmm_chunked(V, which, N_outer, M_aux, chunk=10) -> dict:
    """conditional Monte Carlo MC at V with GMM-IS (banana). Chunked + resumable.

    `which` is "Vtrace" or "Vjhi" and selects both the V column and the
    per-V resume cache so the two banana KLs don't collide.
    """
    seed_outer = SEED_OUTER_TRACE if which == "Vtrace" else SEED_OUTER_HI
    cache = os.path.join(_REPO, f"out/example1_truekl_{which}_partial.json")
    if os.path.exists(cache):
        with open(cache) as f:
            partial = json.load(f)
        inner_kls = list(partial["inner_kls"])
        M_eff_fracs = list(partial["M_eff_fracs"])
        n_done = len(inner_kls)
        print(f"    resuming banana GMM-IS: {n_done}/{N_outer} u's done")
    else:
        inner_kls, M_eff_fracs, n_done = [], [], 0

    U_all = sample_pi_U(N_outer, V, "banana", seed=seed_outer)
    while n_done < N_outer:
        n_next = min(n_done + chunk, N_outer)
        print(f"    chunk {n_done}-{n_next}...", flush=True)
        t0 = time.time()
        res = marginalize_pi_U_with_inner_kl(
            U_all[n_done:n_next], V, "banana",
            M_aux=M_aux, n_mode_starts=20, seed=seed_outer + 1 + n_done,
            inner_sampler="gmm", n_hmc=5000, n_warmup=1500,
            n_leapfrog=40, gmm_K=16,
        )
        inner_kls.extend(float(x) for x in res["inner_kl"])
        M_eff_fracs.extend(float(x) for x in res["M_eff_frac"])
        n_done = n_next
        print(f"      {time.time() - t0:.0f}s; KL mean={np.mean(inner_kls):.4f} "
              f"median={np.median(inner_kls):.4f} "
              f"M_eff median={np.median(M_eff_fracs):.3f}", flush=True)
        with open(cache, "w") as f:
            json.dump({"inner_kls": inner_kls, "M_eff_fracs": M_eff_fracs,
                       "n_done": n_done}, f, indent=2)

    return _summarize_kl(np.asarray(inner_kls, dtype=float), M_aux, {
        "M_eff_frac_median": float(np.median(M_eff_fracs)),
    })


def run_truekl(only: list | None = None) -> dict:
    if not os.path.exists(PARTIAL_PATH):
        raise SystemExit("run --stage bounds first (no out/supplement_table_partial.json)")
    with open(PARTIAL_PATH) as f:
        results = json.load(f)

    labels = only or list(EXAMPLES)
    for label in labels:
        cfg = EXAMPLES[label]
        name = cfg["benchmark"]
        if (results.get(label, {}).get("true_kl_at_Vtrace")
                and results[label].get("true_kl_at_Vjhi")):
            print(f"{label}: true KL already present, skipping")
            continue
        if label not in results:
            raise SystemExit(f"{label} missing from bounds stage; run --stage bounds")

        print(f"\n=== true KL: {label} = {name} "
              f"({cfg['inner_sampler']}) ===", flush=True)
        npz = np.load(os.path.join(_REPO, f"out/V_for_supplement_{label}.npz"))
        V_trace, V_hi = npz["V_trace"], npz["V_jhi"]

        if cfg["inner_sampler"] == "gmm":
            print("  V_trace ...", flush=True)
            kl_tr = _true_kl_gmm_chunked(V_trace, "Vtrace",
                                         cfg["N_outer"], cfg["M_aux"])
            print("  V_hi ...", flush=True)
            kl_hi = _true_kl_gmm_chunked(V_hi, "Vjhi",
                                         cfg["N_outer"], cfg["M_aux"])
        else:
            print("  V_trace ...", flush=True)
            kl_tr = _true_kl_laplace(V_trace, name, cfg["N_outer"],
                                     cfg["M_aux"], SEED_OUTER_TRACE)
            print(f"    KL={kl_tr['true_kl_mean']:.4f} +/- {kl_tr['outer_se']:.4f}")
            print("  V_hi ...", flush=True)
            kl_hi = _true_kl_laplace(V_hi, name, cfg["N_outer"],
                                     cfg["M_aux"], SEED_OUTER_HI)
            print(f"    KL={kl_hi['true_kl_mean']:.4f} +/- {kl_hi['outer_se']:.4f}")

        results[label]["true_kl_at_Vtrace"] = kl_tr
        results[label]["true_kl_at_Vjhi"] = kl_hi
        results[label]["complete"] = True
        with open(PARTIAL_PATH, "w") as f:
            json.dump(results, f, indent=2)

    with open(FINAL_PATH, "w") as f:
        json.dump(results, f, indent=2)
    _mirror_full_to_cache(results)
    print(f"\nWrote {FINAL_PATH}")
    _print_final_table(results)
    return results


def _print_final_table(results: dict) -> None:
    print(f"\n{'Example':<12} {'d':>3} {'r':>3} {'sin th':>8} "
          f"{'J_tr@Vtr':>9} {'J_hi@Vtr':>9} "
          f"{'KL@Vtr':>14} {'KL@Vhi':>14}")
    for label in ["example1", "example2", "example3"]:
        if label not in results:
            continue
        r_ = results[label]
        kt = r_.get("true_kl_at_Vtrace", {})
        kh = r_.get("true_kl_at_Vjhi", {})
        kt_s = (f"{kt.get('true_kl_mean', float('nan')):.3f}"
                f"+/-{kt.get('outer_se', float('nan')):.3f}") if kt else "--"
        kh_s = (f"{kh.get('true_kl_mean', float('nan')):.3f}"
                f"+/-{kh.get('outer_se', float('nan')):.3f}") if kh else "--"
        print(f"{label:<12} {r_['d']:>3} {r_['r']:>3} {r_['sin_theta']:>8.4f} "
              f"{r_['J_trace_at_Vtrace']:>9.4f} {r_['J_hi_at_Vtrace']:>9.4f} "
              f"{kt_s:>14} {kh_s:>14}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["bounds", "truekl", "all"],
                    default="bounds")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset, e.g. example2,example3")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None

    if args.stage in ("bounds", "all"):
        run_bounds(only)
    if args.stage in ("truekl", "all"):
        run_truekl(only)


if __name__ == "__main__":
    main()
