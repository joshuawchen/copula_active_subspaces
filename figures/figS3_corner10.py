"""Corner plots of the first ten coordinates of samples from the true law and
the four estimators. Writes corner10_{oracle,cas,pca,pom,gauss}.png.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, 'src'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cas.config import (K_OUTER, Q_OUTER, K_INNER, Q_INNER_CAP,
                    M_NORM, USE_SECONDARY_RANK, R_ORACLE)
from cas import sample_noise, log_noise_density
from cas import rank_gaussianize
from cas import (ReducedDensityModel,
                              ProductOfMarginalsModel,
                              GaussianCopulaModel,
                              ScalarMarginal)


N_TRAIN = 12500
N_SCATTER = 3000
GRID_RANGE = 5.0
RIDGE = 'theoretical:c=3,delta=1e-5'

D_PLOT = 10  # show first 10 of 20 dimensions

from cas.paperstyle import (
    apply_paper_style,
    COLOR_CAS, COLOR_PCA, COLOR_GAUSS, COLOR_POM, COLOR_TRUE,
)
COLOR_ORACLE  = COLOR_TRUE
COLOR_PRODUCT = COLOR_POM


def _pca_subspace(eta, r):
    Z = rank_gaussianize(eta)
    if isinstance(Z, tuple):
        Z = Z[0]
    Sig = Z.T @ Z / Z.shape[0]
    w, V = np.linalg.eigh(Sig)
    return V[:, np.argsort(-w)[:r]]


def fit_all_models():
    from cas.kq_selection import fit_cas_autoselect
    rng = np.random.default_rng(9000 + N_TRAIN)
    eta_tr = sample_noise(N_TRAIN, rng)
    pom = ProductOfMarginalsModel().fit(eta_tr)
    gauss = GaussianCopulaModel().fit(eta_tr)
    cas_hsm, sel = fit_cas_autoselect(eta_tr, r=R_ORACLE, seed=0)
    print(f"  CAS selected kq=({sel['K1']},{sel['q1']}"
          f"{'e' if sel['even_degree'] else ''})/({sel['K2']},{sel['q2']})",
          flush=True)
    V_pca = _pca_subspace(eta_tr, R_ORACLE)
    pca_hsm = ReducedDensityModel(
        r=R_ORACLE, K_inner=sel["K2"], q_inner=sel["q2"],
        use_secondary_rank=USE_SECONDARY_RANK, M_norm=M_NORM, seed=0,
        inner_ridge_scheme=RIDGE,
    ).fit(eta_tr, V_r_override=V_pca)
    # Reuse pom.marginals — same data, identical fit thanks to ScalarMarginal cache.
    marginals = pom.marginals
    return pom, gauss, pca_hsm, cas_hsm, marginals, eta_tr


def rg_to_eta(z, marginals):
    from scipy.stats import norm
    u = norm.cdf(z)
    eta = np.empty_like(z)
    for i, m in enumerate(marginals):
        eta[:, i] = m.inverse_cdf(u[:, i])
    return eta


def truncation_samples_subspace(eta_oracle, V_r, marginals, rng):
    from scipy.stats import norm
    N, d = eta_oracle.shape
    r = V_r.shape[1]
    z = np.empty_like(eta_oracle)
    for i, m in enumerate(marginals):
        z[:, i] = norm.ppf(m.cdf(eta_oracle[:, i]))
    z_in = (z @ V_r) @ V_r.T
    Q_full, _ = np.linalg.qr(np.hstack([V_r, rng.standard_normal((d, d - r))]))
    V_perp = Q_full[:, r:]
    u_perp_fresh = rng.standard_normal((N, d - r))
    z_perp_fresh = u_perp_fresh @ V_perp.T
    z_tilde = z_in + z_perp_fresh
    return rg_to_eta(z_tilde, marginals)


def truncation_samples_pom(marginals, N, rng):
    z = rng.standard_normal((N, len(marginals)))
    return rg_to_eta(z, marginals)


def truncation_samples_gauss(gauss_model, marginals, N, rng):
    L = np.linalg.cholesky(gauss_model.R + 1e-9 * np.eye(gauss_model.d))
    z = rng.standard_normal((N, gauss_model.d)) @ L.T
    return rg_to_eta(z, marginals)


def make_corner_for(samples, color, name, out_path):
    """Make a single 10-dim corner plot."""
    fig, axes = plt.subplots(D_PLOT, D_PLOT, figsize=(11, 11),
                              sharex='col', sharey='row')

    apply_paper_style()
    plt.rcParams.update({'font.size': 7})

    for ri in range(D_PLOT):
        for ci in range(D_PLOT):
            ax = axes[ri, ci]
            ax.tick_params(labelsize=6)
            if ri == ci:
                ax.hist(samples[:, ri], bins=30, range=(-GRID_RANGE, GRID_RANGE),
                        color=color, alpha=0.6, edgecolor='none')
                ax.set_yticks([])
            elif ri > ci:
                ax.scatter(samples[:, ci], samples[:, ri],
                          s=1.0, color=color, alpha=0.30, edgecolor='none')
                ax.set_xlim(-GRID_RANGE, GRID_RANGE)
                ax.set_ylim(-GRID_RANGE, GRID_RANGE)
            else:
                ax.axis('off')
            if ri == D_PLOT - 1:
                ax.set_xlabel(f'$\\eta_{{{ci}}}$', fontsize=8)
            if ci == 0 and ri != ci:
                ax.set_ylabel(f'$\\eta_{{{ri}}}$', fontsize=8)

    fig.suptitle(f'{name}', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, bbox_inches='tight',
                dpi=int(os.environ.get('CORNER10_DPI', '140')))
    plt.close()
    print(f'[saved] {out_path}')


def main():
    print("Fitting...", flush=True)
    pom, gauss, pca_hsm, cas_hsm, marginals, eta_tr = fit_all_models()

    rng = np.random.default_rng(123)
    oracle_samples = sample_noise(N_SCATTER, rng)

    rng_t = np.random.default_rng(456)
    print("Generating truncation samples...", flush=True)
    pom_samples = truncation_samples_pom(marginals, N_SCATTER, rng_t)
    gauss_samples = truncation_samples_gauss(gauss, marginals, N_SCATTER, rng_t)
    V_pca = _pca_subspace(eta_tr, R_ORACLE)
    pca_samples = truncation_samples_subspace(oracle_samples, V_pca, marginals, rng_t)
    V_cas = cas_hsm.V_r
    cas_samples = truncation_samples_subspace(oracle_samples, V_cas, marginals, rng_t)

    OUT_DIR = os.environ.get('CORNER10_OUT_DIR', HERE)
    print("Building 10D corner plots...", flush=True)
    # Bare-minimum titles: just the method name. Caption-territory text
    # ("first 10 dimensions", N=3000, etc.) stays in the caption.
    make_corner_for(oracle_samples, COLOR_ORACLE, 'True noise',
                    os.path.join(OUT_DIR, 'corner10_oracle.png'))
    make_corner_for(cas_samples, COLOR_CAS, 'CAS',
                    os.path.join(OUT_DIR, 'corner10_cas.png'))
    make_corner_for(pca_samples, COLOR_PCA, 'PCA-HCSM',
                    os.path.join(OUT_DIR, 'corner10_pca.png'))
    make_corner_for(pom_samples, COLOR_PRODUCT, 'PoM',
                    os.path.join(OUT_DIR, 'corner10_pom.png'))
    make_corner_for(gauss_samples, COLOR_GAUSS, 'Gaussian copula',
                    os.path.join(OUT_DIR, 'corner10_gauss.png'))


if __name__ == '__main__':
    main()
