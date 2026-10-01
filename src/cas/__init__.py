"""Copula Active Subspaces: estimator and experiment support for Parts I and II.

Entry points: cas_fit (Stage 1), ReducedDensityModel (the full estimator) and
sample_noise (Example 1). Examples 2 and 3 are in even_fold and
conformal_cube; config holds the default settings.
"""
from __future__ import annotations

# Public API — kept stable for downstream scripts/notebooks
from .hermite import hermite_norm, enumerate_dictionary
from .ranking import rank_gaussianize, ScalarMarginal, rank_transform_through_marginals
from .features import evaluate_features, evaluate_features_psi_only
from .stage1 import cas_fit, cas_fit_streaming, sin_theta_F
from .stage2 import build_A_b, build_A_b_streaming, build_weighted_A_b_streaming
from .ridge import (
    solve_ridge,
    solve_ridge_chaos_block,
    solve_ridge_chaos_eig,
    solve_ridge_full,
    solve_ridge_trace,
    solve_ridge_trace_hybrid,
    solve_ridge_chaos_block_blend,
    solve_ridge_theoretical,
    solve_ridge_hermitescaled,
)
from .densities import (
    ReducedDensityModel,
    ProductOfMarginalsModel,
    GaussianCopulaModel,
    FullDKDEModel,
    eval_g_polynomial,
    clear_scalar_marginal_cache,
)
from .noise import (
    sample_noise, log_noise_density, oracle_subspace, oracle_subspace_geom,
    L_CHOL,
)
from .cubic_banana import (
    sample_cubic_banana_noise,
    log_cubic_banana_density,
    oracle_subspace_cubic_banana,
    D_CB,
    R_CB_TARGET,
    ALPHA_CB,
)
from .ppg import (
    sample_ppg_noise,
    log_ppg_density,
    oracle_subspace_ppg,
    D_PPG,
    R_PPG_TARGET,
    R_PPG_PRIMARY,
    LAMBDAS_PPG,
)
from .even_fold import (
    sample_even_fold_noise,
    log_even_fold_density,
    oracle_subspace_even_fold,
    oracle_subspace_geom_even_fold,
    D_EF,
    R_EF_TARGET,
    ALPHA_EF,
    GAMMA_EF,
)
from .conformal_cube import (
    sample_conformal_cube_noise,
    log_conformal_cube_density,
    oracle_subspace_conformal_cube,
    oracle_subspace_geom_conformal_cube,
    D_C3,
    R_C3_TARGET,
    ALPHA_C3,
    M_C3,
)
from .bip import BIP_A, BIP_XX, posterior_grid, kl, log_prior, log_gauss
from .oracle_reduced import stage1_oracle_kl
from .rank_gauss_density import (
    make_log_pi_Z, fit_marginals_from_eta, rank_gauss_with_marginals,
    compute_population_MHC,
)

__version__ = "0.2.0"
__all__ = [
    # hermite
    "hermite_norm", "enumerate_dictionary",
    # ranking
    "rank_gaussianize", "ScalarMarginal", "rank_transform_through_marginals",
    # features
    "evaluate_features", "evaluate_features_psi_only",
    # stage1
    "cas_fit", "cas_fit_streaming", "sin_theta_F",
    # stage2
    "build_A_b", "build_A_b_streaming", "build_weighted_A_b_streaming",
    # ridge
    "solve_ridge", "solve_ridge_chaos_block", "solve_ridge_chaos_eig",
    "solve_ridge_full", "solve_ridge_trace", "solve_ridge_trace_hybrid",
    "solve_ridge_chaos_block_blend", "solve_ridge_theoretical",
    "solve_ridge_hermitescaled",
    # densities
    "ReducedDensityModel", "ProductOfMarginalsModel", "GaussianCopulaModel",
    "FullDKDEModel", "eval_g_polynomial", "clear_scalar_marginal_cache",
    # noise
    "sample_noise", "log_noise_density", "oracle_subspace",
    "oracle_subspace_geom", "L_CHOL",
    # cubic_banana
    "sample_cubic_banana_noise", "log_cubic_banana_density",
    "oracle_subspace_cubic_banana",
    "D_CB", "R_CB_TARGET", "ALPHA_CB",
    # ppg
    "sample_ppg_noise", "log_ppg_density", "oracle_subspace_ppg",
    "D_PPG", "R_PPG_TARGET", "R_PPG_PRIMARY", "LAMBDAS_PPG",
    # even_fold
    "sample_even_fold_noise", "log_even_fold_density",
    "oracle_subspace_even_fold", "oracle_subspace_geom_even_fold",
    "D_EF", "R_EF_TARGET", "ALPHA_EF", "GAMMA_EF",
    # conformal_cube (z^3)
    "sample_conformal_cube_noise", "log_conformal_cube_density",
    "oracle_subspace_conformal_cube", "oracle_subspace_geom_conformal_cube",
    "D_C3", "R_C3_TARGET", "ALPHA_C3", "M_C3",
    # bip
    "BIP_A", "BIP_XX", "posterior_grid", "kl", "log_prior", "log_gauss",
    # oracle_reduced
    "stage1_oracle_kl",
    # rank_gauss_density
    "make_log_pi_Z", "fit_marginals_from_eta", "rank_gauss_with_marginals",
    "compute_population_MHC",
]
