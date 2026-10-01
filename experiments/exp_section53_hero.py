"""Posterior grids for the observations illustrated in Part I,
fig:bip-posteriors, at N = 50,000. Writes cache/run2026/{example}/hero_N50000.npz.

Usage: python -m experiments.exp_section53_hero --benchmark banana
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cas.config import (
    K_OUTER, Q_OUTER, K_INNER, Q_INNER_CAP,
    M_NORM, USE_SECONDARY_RANK, R_ORACLE, CACHE_DIR, BIP_X_STAR,
)
from cas import (
    sample_noise, log_noise_density, rank_gaussianize,
    ReducedDensityModel, ProductOfMarginalsModel, GaussianCopulaModel,
    BIP_A, BIP_XX, posterior_grid, kl,
    sample_even_fold_noise, log_even_fold_density,
    sample_conformal_cube_noise, log_conformal_cube_density,
)
from cas.kq_selection import fit_cas_autoselect

# Per-benchmark additive noise law (sampler, log-density). One forward map,
# one prior, one truth (BIP_X_STAR) across all three; only the law changes.
_LAWS = {
    "banana":         (sample_noise, log_noise_density),
    "even_fold":      (sample_even_fold_noise, log_even_fold_density),
    "conformal_cube": (sample_conformal_cube_noise, log_conformal_cube_density),
}


# ---------------------------------------------------------------------------
N_TRAIN = 50000
RIDGE = "heldout:c_cov=3,c_sob=1e-5"
X_STAR = np.array(BIP_X_STAR)
HERO_TRIALS = (3, 7, 12, 18, 25, 33)
# ---------------------------------------------------------------------------


def _pca_subspace(eta: np.ndarray, r: int) -> np.ndarray:
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    Sigma = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sigma)
    return V[:, np.argsort(-w)[:r]]


def _atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def run(benchmark: str = "banana") -> None:
    sample_fn, log_density_fn = _LAWS[benchmark]
    out_dir = os.path.join(CACHE_DIR, "run2026", benchmark)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hero_N{N_TRAIN}.npz")

    print(f"=== §5.3 hero  {benchmark}  N={N_TRAIN}, ridge={RIDGE!r} ===",
          flush=True)

    rng = np.random.default_rng(9000 + N_TRAIN)
    eta_tr = sample_fn(N_TRAIN, rng)

    print("Fitting estimators ...", flush=True)
    t0 = time.time(); pom = ProductOfMarginalsModel().fit(eta_tr)
    print(f"  PoM      ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time(); gauss = GaussianCopulaModel().fit(eta_tr)
    print(f"  Gaussian ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    cas, sel = fit_cas_autoselect(eta_tr, r=R_ORACLE, seed=0)
    print(f"  CAS-HCSM  ({time.time() - t0:.0f}s)  selected "
          f"kq=({sel['K1']},{sel['q1']}{'e' if sel['even_degree'] else ''})"
          f"/({sel['K2']},{sel['q2']})", flush=True)
    t0 = time.time()
    V_pca = _pca_subspace(eta_tr, R_ORACLE)
    pca = ReducedDensityModel(
        r=R_ORACLE, K_inner=sel["K2"], q_inner=sel["q2"],
        use_secondary_rank=USE_SECONDARY_RANK, M_norm=M_NORM, seed=0,
        inner_ridge_scheme=RIDGE,
    ).fit(eta_tr, V_r_override=V_pca)
    print(f"  PCA-HCSM  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nGenerating hero-trial posteriors for trials {HERO_TRIALS} ...",
          flush=True)
    trials: dict = {}
    for t in HERO_TRIALS:
        rng_t = np.random.default_rng(20000 + t)
        eta_obs = sample_fn(1, rng_t)[0]
        y = BIP_A @ X_STAR + eta_obs
        residuals = y - BIP_XX @ BIP_A.T

        p_true = posterior_grid(residuals, log_density_fn)
        p_cas = posterior_grid(residuals, cas.evaluate_log_density)
        p_pca = posterior_grid(residuals, pca.evaluate_log_density)
        p_pom = posterior_grid(residuals, pom.evaluate_log_density)
        p_gauss = posterior_grid(residuals, gauss.evaluate_log_density)

        trials[f"{t}"] = dict(
            p_true=p_true, p_cas=p_cas, p_pca=p_pca,
            p_pom=p_pom, p_gauss=p_gauss,
            kl_cas=float(kl(p_true, p_cas)),
            kl_pca=float(kl(p_true, p_pca)),
            kl_pom=float(kl(p_true, p_pom)),
            kl_gauss=float(kl(p_true, p_gauss)),
            x_star=X_STAR.copy(),
            eta_obs=eta_obs,
        )
        print(f"  trial {t:>3} cas={trials[str(t)]['kl_cas']:.2f} "
              f"pca={trials[str(t)]['kl_pca']:.2f} "
              f"pom={trials[str(t)]['kl_pom']:.2f} "
              f"gauss={trials[str(t)]['kl_gauss']:.2f}", flush=True)

    _atomic_save_npz(out_path, trials=trials)
    print(f"\nSaved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="§5.3 hero-trial BIP posteriors")
    p.add_argument("--benchmark", default="banana", choices=list(_LAWS),
                   help="additive noise law (Example 1/2/3)")
    args = p.parse_args()
    run(args.benchmark)


if __name__ == "__main__":
    main()
