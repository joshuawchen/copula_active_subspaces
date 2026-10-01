"""Main experiments of Part I, Section 5. For each example, N and seed: one fit
with (K, q) selected on the cell's own sample (fit_cas_autoselect), and the
subspace angles, noise KL divergences and posterior KL divergences of CAS,
PCA-HCSM, the product of marginals and the Gaussian copula.
Writes cache/run2026/{example}/{subspace,testll,posterior}_N{N}.json; resumable.

Smoke run: PAPER_RUN_EXAMPLES=banana PAPER_RUN_NS=500 PAPER_RUN_SEEDS=1
python -m experiments.exp_paper_run
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from cas.config import (CACHE_DIR, R_ORACLE, BIP_X_STAR, M_NORM,
                         USE_SECONDARY_RANK, RIDGE_DEFAULT)
from cas import (
    sample_noise, log_noise_density,
    sample_even_fold_noise, log_even_fold_density,
    sample_conformal_cube_noise, log_conformal_cube_density,
    rank_gaussianize, sin_theta_F,
    ReducedDensityModel, ProductOfMarginalsModel, GaussianCopulaModel,
    BIP_A, BIP_XX, posterior_grid, kl,
)
from cas.kq_selection import fit_cas_autoselect
from cas.analytic_oracle import analytic_oracle_subspace

_LAWS = {
    "banana":         (sample_noise, log_noise_density),
    "even_fold":      (sample_even_fold_noise, log_even_fold_density),
    "conformal_cube": (sample_conformal_cube_noise, log_conformal_cube_density),
}

R = R_ORACLE
X_STAR = np.array(BIP_X_STAR)
_RUN_DIR = os.path.join(CACHE_DIR, "run2026")
# One ridge scheme for both rank-r methods, by reference to the config default
# that fit_cas_autoselect uses for CAS.
_RIDGE = RIDGE_DEFAULT

_ns_env = os.environ.get("PAPER_RUN_NS", "")
NS = ([int(x) for x in _ns_env.split(",")] if _ns_env
      else [500, 1000, 2000, 5000, 10000, 20000, 50000])
SEEDS = int(os.environ.get("PAPER_RUN_SEEDS", "20"))
_ex_env = os.environ.get("PAPER_RUN_EXAMPLES", "")
EXAMPLES = (_ex_env.split(",") if _ex_env else list(_LAWS))


def _pca_subspace(eta, r):
    Z = rank_gaussianize(eta)
    Sigma = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sigma)
    return V[:, np.argsort(-w)[:r]]


def _random_subspace(d, r, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return Q[:, :r]


def _cache_path(example, quantity, N):
    d = os.path.join(_RUN_DIR, example)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{quantity}_N{N}.json")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _noise_kl(model, eta_te, log_density_fn):
    ll_true = float(np.mean(log_density_fn(eta_te)))
    ll_est = float(np.mean(model.evaluate_log_density(eta_te)))
    return ll_true - ll_est


def _posterior_kl(model, residuals, p_oracle):
    try:
        p_model = posterior_grid(residuals, model.evaluate_log_density)
        return float(kl(p_oracle, p_model))
    except Exception:
        return float("nan")


def run_cell(example, N, seed):
    """One fit per method on a seed's sample; return (subspace, testll,
    posterior, selected_kq). All three metric dicts come from the same fits."""
    sample_fn, log_density_fn = _LAWS[example]
    rng = np.random.default_rng(7000 + seed)
    eta_tr = sample_fn(N, rng)
    eta_te = sample_fn(N, rng)
    Vref = analytic_oracle_subspace(example, R)
    d = eta_tr.shape[1]

    # CAS: fully data-driven -- selects both stages on eta_tr, refits on eta_tr.
    cas, sel = fit_cas_autoselect(eta_tr, r=R, seed=seed)

    # PCA-HCSM: PCA subspace, same auto-selected Stage-2 (K2,q2), and every
    # other setting pinned to the CAS value (M_NORM, USE_SECONDARY_RANK,
    # RIDGE_DEFAULT), so the two differ in the subspace alone.
    V_pca = _pca_subspace(eta_tr, R)
    pca = ReducedDensityModel(
        r=R, K_inner=sel["K2"], q_inner=sel["q2"], seed=seed,
        inner_ridge_scheme=_RIDGE, M_norm=M_NORM,
        use_secondary_rank=USE_SECONDARY_RANK,
    ).fit(eta_tr, V_r_override=V_pca)

    pom = ProductOfMarginalsModel().fit(eta_tr)
    gauss = GaussianCopulaModel().fit(eta_tr)

    # Subspace sinThetaF vs the exact V_r^C (diagnostic; not used by the fit).
    V_rand = _random_subspace(d, R, rng)
    subspace = {
        "CAS": float(sin_theta_F(cas.V_r, Vref)),
        "PCA": float(sin_theta_F(V_pca, Vref)),
        "random": float(sin_theta_F(V_rand, Vref)),
    }

    # Noise KL (held-out): E[log pi] - E[log hat_pi].
    testll = {
        "CAS": _noise_kl(cas, eta_te, log_density_fn),
        "PCA": _noise_kl(pca, eta_te, log_density_fn),
        "PoM": _noise_kl(pom, eta_te, log_density_fn),
        "Gaussian": _noise_kl(gauss, eta_te, log_density_fn),
    }

    # BIP posterior KL: ONE observation, tied to the seed (no obs explosion).
    rng_obs = np.random.default_rng(20000 + seed)
    eta_obs = sample_fn(1, rng_obs)[0]
    y = BIP_A @ X_STAR + eta_obs
    residuals = y - BIP_XX @ BIP_A.T
    p_oracle = posterior_grid(residuals, log_density_fn)
    posterior = {
        "CAS": _posterior_kl(cas, residuals, p_oracle),
        "PCA": _posterior_kl(pca, residuals, p_oracle),
        "PoM": _posterior_kl(pom, residuals, p_oracle),
        "Gaussian": _posterior_kl(gauss, residuals, p_oracle),
    }
    return subspace, testll, posterior, sel


def run_one_N(example, N, n_seeds):
    sp_path = _cache_path(example, "subspace", N)
    tl_path = _cache_path(example, "testll", N)
    po_path = _cache_path(example, "posterior", N)
    sp, tl, po = _load(sp_path), _load(tl_path), _load(po_path)
    for cache in (sp, tl, po):
        for m in ("CAS", "PCA", "PoM", "Gaussian", "random"):
            cache.setdefault(m, {})
    sp.setdefault("selected_kq", {})

    print(f"=== {example}  N={N}  ({n_seeds} seeds) ===", flush=True)
    for seed in range(n_seeds):
        s = str(seed)
        if s in sp["CAS"] and s in tl["CAS"] and s in po["CAS"]:
            continue
        t0 = time.time()
        subspace, testll, posterior, sel = run_cell(example, N, seed)
        for m, v in subspace.items():
            sp[m][s] = v
        sp["selected_kq"][s] = sel
        for m, v in testll.items():
            tl[m][s] = v
        for m, v in posterior.items():
            po[m][s] = v
        _save(sp_path, sp)
        _save(tl_path, tl)
        _save(po_path, po)
        ev = "e" if sel["even_degree"] else ""
        print(f"  seed {seed + 1}/{n_seeds} ({time.time() - t0:.0f}s)  "
              f"kq=({sel['K1']},{sel['q1']}{ev})/({sel['K2']},{sel['q2']})  "
              f"sinTheta[CAS]={subspace['CAS']:.3f}  "
              f"noiseKL[CAS]={testll['CAS']:.3f}  postKL[CAS]={posterior['CAS']:.3f}",
              flush=True)


def main():
    print(f"paper run: examples={EXAMPLES}  N={NS}  seeds={SEEDS}")
    print(f"cache dir: {_RUN_DIR}", flush=True)
    for example in EXAMPLES:
        for N in NS:
            run_one_N(example, N, SEEDS)
    print("=== paper run complete ===")


if __name__ == "__main__":
    main()
