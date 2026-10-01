"""Default settings: paths, parallelism, Hermite truncation, regularizer
constants, normalizer budgets and the forward map of the inference problem.
"""
from __future__ import annotations
import os


# ---------------------------------------------------------------------------
# 1. Filesystem
# ---------------------------------------------------------------------------
# Repo root is two levels up from this file (src/cas/config.py).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Parallelism
# ---------------------------------------------------------------------------
# Worker count for embarrassingly-parallel sweeps. None => max(1, cores - 1).
# BLAS thread count is pinned to 1 per worker to prevent oversubscription;
# see cas.parallel.run_parallel_init for the spawn-safe initializer.
# Override at runtime with the CAS_N_WORKERS env var (reproduce.py sets it
# from --workers to cap the outer pool so it cannot oversubscribe RAM).
N_WORKERS: int | None = (int(os.environ["CAS_N_WORKERS"])
                         if os.environ.get("CAS_N_WORKERS") else None)


# ---------------------------------------------------------------------------
# 3. Noise law (§5: 20-D banana copula)
# ---------------------------------------------------------------------------
# Eq (5.1) — Gaussian copula warped by per-block "banana" map z2 += a*(z1^2 - 1)
# with logistic marginals. K_BAN coupled blocks of dimension 2; the remaining
# d - 2*K_BAN dimensions are independent logistic.
D_OBS: int = 20      # ambient dimension
K_BAN: int = 2       # number of coupled banana blocks
A_BAN: float = 0.7   # banana curvature parameter
RHO: float = 0.2     # Toeplitz correlation between block coordinates
R_ORACLE: int = 2 * K_BAN  # = 4, oracle subspace rank V_r^*


# ---------------------------------------------------------------------------
# 4. Hermite-polynomial truncation (§3.3 outer / §3.4 inner)
# ---------------------------------------------------------------------------
# Outer dictionary Λ_outer (§3.3): tensor-product Hermite basis on R^d, total
# degree ≤ K_OUTER, interaction order ≤ Q_OUTER. This is the basis on which
# the copula-score covariance C is estimated.
K_OUTER: int = 4
Q_OUTER: int = 2

# Inner dictionary Λ_inner (§3.4): basis on R^r used for the polynomial
# log-density on the reduced coordinates. The actual interaction order used
# is min(R_ORACLE, Q_INNER_CAP) so the cap kicks in only when r > 3.
K_INNER: int = 5
Q_INNER_CAP: int = 3

# Secondary rank-Gaussianization of U = V_r^T Z before the inner fit.
# False = paper convention (fit directly on U; matches Eqs 3.5, 3.7).
# True  = legacy variant with an extra componentwise rank transform on U
#         and a Jacobian correction at evaluation. Empirically within seed
#         noise on the banana benchmark.
USE_SECONDARY_RANK: bool = False


# ---------------------------------------------------------------------------
# 4b. (K,q) selection is per-sample and data-driven (no fixed per-example config)
# ---------------------------------------------------------------------------
# (K1,q1) and (K2,q2) are NOT fixed here. The deployed estimator selects them
# per sample, by held-out score matching on an in/out split of the training
# sample itself (cas.kq_selection.fit_cas_autoselect), using only the data
# available at that N -- no pilot and no oracle. At small N this selects a
# leaner dictionary (as the sample supports); the selection sharpens with N, and
# CAS beats the baselines at every N regardless. The candidate grids and the
# per-sample selector live in cas.kq_selection. The globals above (K_OUTER
# etc.) remain the defaults for legacy single-law callers.


# ---------------------------------------------------------------------------
# 5. Regularizers (§4)
# ---------------------------------------------------------------------------
# Published default for every §5.2 / §5.3 number is
#   inner_ridge_scheme = "heldout:c_cov=3,c_sob=1e-5"
# (legacy form "theoretical:c=3,delta=1e-5" still accepted; c↔c_cov,
# delta↔c_sob), which dispatches to cas.ridge.solve_ridge_theoretical
# (paper §3.3, Eq. R-combined). That solver implements R = R_cov + R_sob
# with a centering constraint:
#
#     R_cov = c_cov · diag_k( σ̂²_k I_{p_k} )       σ̂²_k := tr(Â_r^{(k,k)}) / N
#     R_sob = c_sob · ‖Â_r‖_op · diag(|α|₁²)
#     constraint:  Ĉ θ = 0,  Ĉ[j,α] = (1/N) Σ_n ∂_j H_α(u_n)
#
# Per the paper (Eq. R-combined), R_cov and R_sob are formally separate
# block-diagonal pieces: R_cov is degree-class identities I_{p_k}
# scaled by σ̂²_k; R_sob is a global diagonal in |α|₁². On each class they
# add to a per-class identity with scalar (c_cov · σ̂²_k + c_sob · ‖Â_r‖_op · k²).
#
# Production default: 'heldout' selects the two global scalars per fit by
# out-of-sample score-matching risk (Part I supplement, sec:sm2-selection);
# the (c_cov, c_sob) in the string are the grid CENTERS, with representative
# values (3.0, 1e-5). The fixed-pair form 'theoretical:c_cov=3,c_sob=1e-5'
# remains for ridge ablations and probes that pin the pair by design.
RIDGE_DEFAULT: str = "heldout:c_cov=3,c_sob=1e-5"

# Sup-constrained Stage-2 refinement. When True, the final full-sample Stage-2
# refit is re-solved over the convex class defined by the level bound
#     L_beta(u) <= tau on the ball ||u|| <= rmax,   tau = 99.9th in-sample
# percentile of the unconstrained potential, and the leading-form margin
# F_K[beta](v) <= -eps on the unit sphere, via the exchange solve of
# cas.constrained (eps = 1e-3; rmax = max(18, 1.5 * max ||u||)). The reduced
# factor is then normalizable: no evaluation-time level bound, and the
# normalizer integrates the potential as is. Held-out selection inner
# fits stay unconstrained; only theoretical/heldout schemes are
# refined (ablation schemes keep their unconstrained solves).
# Override per model by setting the attribute constrained_solve.
CONSTRAINED_DEFAULT: bool = True

# Trace-hybrid regularizer R_tr (Eq 4.5, an earlier / alternative scheme of
# Theorem 4.3, kept available for §4 ablations only — not the published default).
TR_C: float = 3.0       # inverse SNR
TR_DELTA: float = 1e-5  # inverse signal-to-misspecification floor
TR_P: float = 4.0       # sigmoid steepness

# Sobolev R_{H^2} ridge for the comparison study in Fig 2 / Table 1.
RH2_KAPPA_DEFAULT: float = 1e-3
RH2_KAPPA_GRID: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)


# ---------------------------------------------------------------------------
# 6. Normalizer estimation (§3.5, Appendix H)
# ---------------------------------------------------------------------------
# Scrambled-Sobol budget M for log Z_r ≈ log E_{γ_r}[exp(g̃)]. Used only at
# evaluation time for held-out log-likelihood; not in the fit pipeline. The
# posterior KL is unaffected by it: the grid renormalizes and the constant
# cancels.
M_NORM: int = 131072

# Scrambled-Sobol budget for the composed normalizer Z_n (estimate_log_mass).
M_MASS: int = 131072


# ---------------------------------------------------------------------------
# 7. Experiment configs (§5)
# ---------------------------------------------------------------------------
# Each EXP_* dict carries the (cache_path, N_train, n_seeds/n_trials, ...)
# tuple specifying a published experiment. See experiments/exp_*.py.

# §5.1 Subspace recovery (Fig 3 left panel)
EXP_SUBSPACE = dict(
    cache=os.path.join(CACHE_DIR, "subspace.npz"),
    N_train=50000,
    n_seeds=20,
    rank=R_ORACLE,
    K_q_pairs=((2, 2), (3, 3), (4, 2), (5, 2)),
)

# §5.2 Held-out test log-likelihood (Fig 4)
EXP_TESTLL = dict(
    cache=os.path.join(CACHE_DIR, "testll.npz"),
    N_train=50000,
    N_test=50000,
    n_seeds=20,
    rank=R_ORACLE,
    methods=("PoM", "Gaussian", "PCA", "CAS"),
)

# §5.3a BIP posterior KL at fixed N (Fig 5, Fig 6 left)
EXP_BIP_FIXEDN = dict(
    cache=os.path.join(CACHE_DIR, "bip_fixedN.npz"),
    N_train=50000,
    n_trials=50,
    rank=R_ORACLE,
    methods=("PoM", "Gaussian", "PCA", "CAS"),
)

# §5.3b BIP posterior KL versus N (Fig 6 right)
EXP_BIP_NSCAN = dict(
    cache=os.path.join(CACHE_DIR, "bip_Nscan.npz"),
    N_train=(100, 500, 2500, 12500, 50000),
    n_trials=30,
    rank=R_ORACLE,
    methods=("PoM", "Gaussian", "PCA", "CAS"),
)

# Table 3.1: regularizer-κ grid (R_{H^2} comparison)
EXP_KAPPA_GRID = dict(
    cache=os.path.join(CACHE_DIR, "kappa_grid.npz"),
    N_train=(100, 500, 2500, 12500, 50000),
    n_trials=50,
    rank=R_ORACLE,
    kappa_grid=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
)

# §4 Fig 2 ridge calibration
EXP_RIDGE = dict(
    cache=os.path.join(CACHE_DIR, "ridge.npz"),
    N_train=(100, 500, 2500, 12500, 50000),
    n_trials_invariance=100,
    rank=R_ORACLE,
)


# ---------------------------------------------------------------------------
# 8. BIP forward map (§5.3)
# ---------------------------------------------------------------------------
# Linear 2-D BIP: y = A x + η, η ~ noise law (§3). Prior is Gaussian on x
# with std = BIP_SIGMA_PRIOR. The matrix A and Cholesky factor are built in
# cas.bip from these parameters.
BIP_SIGMA_PRIOR: float = 2.0
BIP_ALPHA_SIGNAL: float = 1.5   # forward-map scale on the unit-normalized conformal input directions (§5.3, z^2 law)
BIP_GRID_MIN: float = -6.0
BIP_GRID_MAX: float = 6.0
BIP_GRID_N: int = 80
# Shared BIP truth x* for the §5.3 hero/KL experiments (and the SM1/SM3 BIP
# experiments). One forward map, one prior, one truth across all three noise
# laws (banana z^2, even_fold, conformal z^3); only the additive noise law
# changes between examples.
#
# Was (-3, 3), inherited from the Toeplitz-mixed laws.  Under the orthogonal
# mixing that point places 2.2% of the posterior mass beyond 0.85 * BIP_GRID_MAX,
# so the reference posterior itself is truncated by the grid and every KL is then
# computed against a clipped reference.  A scan over x* by the separation
# kl_gauss - kl_cas ranks the boundary of the box highest, but that ranking is an
# artifact: at (5, -3) the clipped mass is 18%, and the scan compares clipped
# against clipped, so it cannot see the truncation it is rewarding.
#
# (-2, 2) is 1.4 sigma under the N(0, BIP_SIGMA_PRIOR^2) prior (so it is a
# typical truth, not a tail draw), leaves 0.02% of the mass beyond 0.85 * grid
# (an intact reference), keeps the posterior clearly bimodal (mode masses
# 0.61 / 0.24), and separates the estimators: on Example 1 at N = 5e4 the
# Gaussian-copula posterior KL is 4.87x that of Cas at r = 4 and 15.5x at r = 8.
BIP_X_STAR: tuple[float, float] = (-2.0, 2.0)
