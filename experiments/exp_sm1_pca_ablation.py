"""PCA subspace baseline of the Part I supplement: tab:pca-banana (held-out
log-likelihood for r in {2, 3, 4, 6, 8} and N in {500, 1000, 2000}) and
tab:pca-bip (posterior KL over 30 observations).
Writes cache/sm1_pca_banana.json and cache/sm1_pca_bip.json; resumable.

Usage: python3 experiments/exp_sm1_pca_ablation.py --stage all --n-seeds 20 --n-trials 30
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

from cas.config import (
    K_OUTER, Q_OUTER, K_INNER, Q_INNER_CAP,
    M_NORM, USE_SECONDARY_RANK, CACHE_DIR, BIP_X_STAR,
)
from cas import (
    sample_noise, log_noise_density, rank_gaussianize,
    ReducedDensityModel, ProductOfMarginalsModel, GaussianCopulaModel,
    BIP_A, BIP_XX, posterior_grid, kl,
)

RIDGE = "heldout:c_cov=3,c_sob=1e-5"
R_VALUES = (2, 3, 4, 6, 8)
N_VALUES_BANANA = (500, 1000, 2000)
SPLIT = 0.8                       # 80/20 train/test (SM7 protocol)
N_SEEDS_DEFAULT = 20
SEED_BASE = 7000                  # matches exp_section52 seeding convention

# BIP stage
X_STAR = np.array(BIP_X_STAR)
N_BIP = 50000                     # training size for the BIP estimators
N_TRIALS_DEFAULT = 30
BANANA_CACHE = os.path.join(CACHE_DIR, "sm1_pca_banana.json")
BIP_CACHE = os.path.join(CACHE_DIR, "sm1_pca_bip.json")
LOG_DIR = os.path.join(os.path.dirname(HERE), "out", "logs")


def _atomic(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _pca_subspace(eta: np.ndarray, r: int) -> np.ndarray:
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    Sigma = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sigma)
    return V[:, np.argsort(-w)[:r]]


def _common_kwargs(r: int, seed: int) -> dict:
    return dict(
        r=r, K=K_OUTER, q=Q_OUTER, K_inner=K_INNER,
        q_inner=min(r, Q_INNER_CAP), use_secondary_rank=USE_SECONDARY_RANK,
        M_norm=M_NORM, seed=seed, inner_ridge_scheme=RIDGE,
    )


# ===========================================================================
# Stage: banana test-LL r-scan (tab:pca-banana)
# ===========================================================================

def _banana_cell(r: int, N: int, seed: int) -> dict:
    """One (r, N, seed): 80/20 split, CAS-HCSM + PCA-HCSM test LL on Example 1."""
    rng = np.random.default_rng(SEED_BASE + seed)
    eta = sample_noise(N, rng)
    n_tr = int(SPLIT * N)
    tr, te = eta[:n_tr], eta[n_tr:]
    kw = _common_kwargs(r, seed)
    cas = ReducedDensityModel(**kw).fit(tr)
    pca = ReducedDensityModel(**kw).fit(tr, V_r_override=_pca_subspace(tr, r))
    return {
        "cas": float(np.mean(cas.evaluate_log_density(te))),
        "pca": float(np.mean(pca.evaluate_log_density(te))),
        "unmet": _level_unmet(cas) + _level_unmet(pca),
    }


def _level_unmet(model) -> int:
    """1 if the model's constrained Stage-2 solve ended at its round cap with
    the level bound unmet (the last iterate is kept), else 0."""
    d = getattr(model, "constrained_diag", None)
    return int(bool(d) and not d.get("level_met", True))


def run_banana(n_seeds: int, only_r=None, only_N=None) -> dict:
    data = json.load(open(BANANA_CACHE)) if os.path.exists(BANANA_CACHE) else {}
    rs = [only_r] if only_r else list(R_VALUES)
    Ns = [only_N] if only_N else list(N_VALUES_BANANA)
    t0 = time.time()
    level = {}   # r -> [fits with the level bound unmet, fits], over this run
    for r in rs:
        data.setdefault(str(r), {})
        for N in Ns:
            cell = data[str(r)].setdefault(str(N), {"cas": [], "pca": []})
            done = len(cell["cas"])
            for seed in range(done, n_seeds):
                res = _banana_cell(r, N, seed)
                cell["cas"].append(res["cas"])
                cell["pca"].append(res["pca"])
                count = level.setdefault(r, [0, 0])
                count[0] += res["unmet"]
                count[1] += 2
                _atomic(BANANA_CACHE, data)
            print(f"  r={r} N={N}: {len(cell['cas'])}/{n_seeds} seeds "
                  f"(CAS {np.mean(cell['cas']):.2f}, PCA {np.mean(cell['pca']):.2f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    if level:
        lines = [f"rank {r}: Stage-2 solve ended at the round cap with the level "
                 f"bound unmet in {u} of the {n} fits computed in this run"
                 for r, (u, n) in sorted(level.items())]
        print("\n" + "\n".join(lines), flush=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "sm1_level_bound.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return data


def summarize_banana(data: dict) -> None:
    """Print the table cells + the caption/Reading statistics."""
    print("\n=== tab:pca-banana (mean test LL) ===")
    hdr = "  r |" + "".join(f"  N={N:<5} CAS/PCA" for N in N_VALUES_BANANA)
    print(hdr)
    wins = ties = losses = 0
    gap_r4 = []
    for r in R_VALUES:
        line = f"  {r} |"
        for N in N_VALUES_BANANA:
            cell = data.get(str(r), {}).get(str(N))
            if not cell or not cell["cas"]:
                line += "      --        "
                continue
            c, p = np.mean(cell["cas"]), np.mean(cell["pca"])
            line += f"  {c:7.2f}/{p:7.2f}"
            if c - p > 0.05:
                wins += 1
            elif c - p < -0.05:
                losses += 1
            else:
                ties += 1
            if r == 4 and N >= 1000:
                gap_r4.append(c - p)
        print(line)
    print(f"\n  caption stats: CAS wins {wins}/{wins+ties+losses}, ties {ties}, "
          f"loses {losses}")
    if gap_r4:
        print(f"  Reading: r=4,N>=1000 gap (CAS-PCA) = {np.mean(gap_r4):.2f} nats")
    # PCA plateau + Gaussian-copula reference printed if present
    pca_vals = [np.mean(data[str(r)][str(N)]["pca"])
                for r in R_VALUES for N in N_VALUES_BANANA
                if data.get(str(r), {}).get(str(N), {}).get("pca")]
    if pca_vals:
        print(f"  PCA plateau ~ {np.median(pca_vals):.2f} nats")


# ===========================================================================
# Stage: BIP posterior-KL r-scan (tab:pca-bip)
# ===========================================================================

def _fit_bip_estimators(N: int) -> dict:
    rng = np.random.default_rng(9000 + N)
    eta_tr = sample_noise(N, rng)
    fitted = {
        "Gaussian": GaussianCopulaModel().fit(eta_tr),
        "PoM": ProductOfMarginalsModel().fit(eta_tr),
    }
    V_cache = {}
    for r in R_VALUES:
        kw = _common_kwargs(r, 0)
        fitted[f"PCA r={r}"] = ReducedDensityModel(**kw).fit(
            eta_tr, V_r_override=_pca_subspace(eta_tr, r))
    fitted["CAS r=4"] = ReducedDensityModel(**_common_kwargs(4, 0)).fit(eta_tr)
    return fitted


def _bip_trial(trial_idx: int, fitted: dict) -> dict:
    rng_t = np.random.default_rng(20000 + trial_idx)
    eta_obs = sample_noise(1, rng_t)[0]
    y = BIP_A @ X_STAR + eta_obs
    residuals = y - BIP_XX @ BIP_A.T
    p_oracle = posterior_grid(residuals, log_noise_density)
    out = {}
    for name, model in fitted.items():
        try:
            out[name] = float(kl(p_oracle, posterior_grid(
                residuals, model.evaluate_log_density)))
        except Exception:
            out[name] = float("nan")
    return out


def run_bip(n_trials: int) -> dict:
    data = json.load(open(BIP_CACHE)) if os.path.exists(BIP_CACHE) else {}
    fitted = _fit_bip_estimators(N_BIP)
    for name in fitted:
        data.setdefault(name, [])
    start = min(len(data[m]) for m in fitted)
    t0 = time.time()
    for t in range(start, n_trials):
        kls = _bip_trial(t, fitted)
        for name, v in kls.items():
            data[name].append(v)
        _atomic(BIP_CACHE, data)
        if (t + 1) % 5 == 0:
            print(f"  trial {t+1}/{n_trials} [{time.time()-t0:.0f}s]", flush=True)
    return data


def summarize_bip(data: dict) -> None:
    print("\n=== tab:pca-bip (posterior KL over trials) ===")
    def stats(name):
        a = np.array([v for v in data.get(name, []) if not np.isnan(v)])
        return (float(a.mean()), float(np.median(a)), float(a.max())) if len(a) else (np.nan,)*3
    g_mean = stats("Gaussian")[0]
    print(f"  {'method':<14} {'mean':>6} {'median':>7} {'worst':>6} {'red. vs Gauss':>14}")
    for name in ["Gaussian", "PoM"] + [f"PCA r={r}" for r in R_VALUES] + ["CAS r=4"]:
        m, md, w = stats(name)
        red = "---" if name == "Gaussian" else f"{100*(g_mean-m)/g_mean:+.0f}%"
        print(f"  {name:<14} {m:>6.2f} {md:>7.2f} {w:>6.2f} {red:>14}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["banana", "bip", "all"], default="banana")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--only-r", type=int, default=None)
    ap.add_argument("--only-N", type=int, default=None)
    args = ap.parse_args()

    if args.stage in ("banana", "all"):
        print(f"=== SM1 banana test-LL r-scan ({args.n_seeds} seeds) ===", flush=True)
        summarize_banana(run_banana(args.n_seeds, args.only_r, args.only_N))
    if args.stage in ("bip", "all"):
        print(f"=== SM1 BIP KL r-scan ({args.n_trials} trials, N={N_BIP}) ===", flush=True)
        summarize_bip(run_bip(args.n_trials))


if __name__ == "__main__":
    main()
