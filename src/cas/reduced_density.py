"""The composed CAS estimator of Part I, eq:cas-estimator, and its baselines.

ReducedDensityModel fits the Stage-1 subspace and the Stage-2 reduced factor
and evaluates

    log pi_hat(x) = L_r(V_r^T z(x)) + sum_i log f_i(x_i) - log Z_n,

with z(x) = Phi^{-1}(F_hat(x)) the empirical rank transform, f_i the kernel
marginals and Z_n estimated by estimate_log_mass. The product-of-marginals
and Gaussian-copula baselines are defined here as well.
"""

import os
import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from .hermite_score_matching import (
    hermite_norm,
    enumerate_dictionary,
    evaluate_features,
    cas_fit_streaming,
    rank_gaussianize,
)


# ---------------------------------------------------------------
# ScalarMarginal cache: when the same fit-data column is used to construct
# multiple model classes (PoM, GaussianCopula, ReducedDensityModel, ...) on the
# same training data, fitting 20 marginals 4× is wasteful — each
# ScalarMarginal.__init__ does a 5000-point KDE evaluation costing ~1s for
# N=12500. Memoize on (id(buffer), shape, dtype, sha256-of-bytes-prefix).
# Cache is process-local; cleared on process restart.
# ---------------------------------------------------------------

_SCALAR_MARGINAL_CACHE = {}      # key -> ScalarMarginal instance
_SCALAR_MARGINAL_CACHE_MAXSIZE = 256  # bound memory; LRU-style eviction


def _scalar_marginal_cache_key(x):
    """Build a content-addressed key for the input array.

    Uses (shape, dtype.str, hash of contiguous bytes). Hashing 100k float64s
    via hashlib.blake2b takes ~50us, vs ~1s for a fresh ScalarMarginal fit at
    N=12500, so the cache key cost is negligible.
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=float).ravel())
    import hashlib
    h = hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()
    return (a.shape, a.dtype.str, h)


def clear_scalar_marginal_cache():
    """Clear the process-local ScalarMarginal cache."""
    _SCALAR_MARGINAL_CACHE.clear()


def _build_marginal_cached(x):
    """Return a ScalarMarginal for x, hitting the process-local cache when
    the same array bytes have been seen before. Safe to use across model
    classes (PoM, GaussianCopula, ReducedDensityModel) — the resulting
    object is read-only with respect to the data it was fitted on, so
    sharing instances is correct.
    """
    key = _scalar_marginal_cache_key(x)
    cached = _SCALAR_MARGINAL_CACHE.get(key)
    if cached is not None:
        return cached
    m = ScalarMarginal(x)  # bypass cache; full fit
    if len(_SCALAR_MARGINAL_CACHE) >= _SCALAR_MARGINAL_CACHE_MAXSIZE:
        # Drop oldest (insertion-order) entry. dict preserves insertion order
        # in CPython 3.7+, so this is effectively FIFO eviction.
        _SCALAR_MARGINAL_CACHE.pop(next(iter(_SCALAR_MARGINAL_CACHE)))
    _SCALAR_MARGINAL_CACHE[key] = m
    return m


def _build_marginals_cached(X):
    """Build per-column ScalarMarginals for matrix X, using the cache.

    Cold-start (no cache hits) on N>>1 is dominated by per-column KDE grid
    evaluation (~1s each at N=12500). When SCALAR_MARGINAL_PARALLEL=1 is set,
    cold builds run in a ThreadPoolExecutor; on multi-core systems where the
    underlying SciPy KDE releases the GIL, this gives near-linear speedup.
    On single-core systems or when GIL is held, parallelism is a no-op so
    correctness is unaffected.

    When two columns of X have identical bytes (rare but possible — e.g.
    duplicated columns or a tiled test matrix), only one ScalarMarginal is
    built and shared.
    """
    d = X.shape[1]
    # Probe the cache: which columns are already cached?
    cached_keys = [_scalar_marginal_cache_key(X[:, i]) for i in range(d)]
    hits = [_SCALAR_MARGINAL_CACHE.get(k) for k in cached_keys]

    # Identify columns that need a build, deduping by cache key so identical
    # columns share a single fit.
    seen_misses: dict = {}
    miss_cols: list[int] = []
    for i, h in enumerate(hits):
        if h is None and cached_keys[i] not in seen_misses:
            seen_misses[cached_keys[i]] = i
            miss_cols.append(i)

    if not miss_cols:
        # Either all hit or all duplicates of cached entries (handled below)
        for i, h in enumerate(hits):
            if h is None:
                hits[i] = _SCALAR_MARGINAL_CACHE.get(cached_keys[i])
        return hits

    # Cold builds for the unique-key misses
    use_parallel = (os.environ.get('SCALAR_MARGINAL_PARALLEL', '0') == '1'
                    and len(miss_cols) > 1)
    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor
        n_workers = min(len(miss_cols), int(os.environ.get(
            'SCALAR_MARGINAL_WORKERS', max(1, (os.cpu_count() or 2) - 1))))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            built = list(ex.map(lambda i: ScalarMarginal(X[:, i]), miss_cols))
    else:
        built = [ScalarMarginal(X[:, i]) for i in miss_cols]

    # Insert each unique build into the cache, then assign hits by key
    for i, m in zip(miss_cols, built):
        if len(_SCALAR_MARGINAL_CACHE) >= _SCALAR_MARGINAL_CACHE_MAXSIZE:
            _SCALAR_MARGINAL_CACHE.pop(next(iter(_SCALAR_MARGINAL_CACHE)))
        _SCALAR_MARGINAL_CACHE[cached_keys[i]] = m

    # Now fill all positions (including duplicate keys) from the cache
    for i in range(d):
        if hits[i] is None:
            hits[i] = _SCALAR_MARGINAL_CACHE[cached_keys[i]]
    return hits


# ---------------------------------------------------------------
# Scalar marginal fits: empirical CDF with smoothed PDF via KDE
# ---------------------------------------------------------------

def _silverman_robust_factor(x):
    """Silverman's robust rule-of-thumb bandwidth factor for gaussian_kde.

    Returns h/sigma, where:
        h = 0.9 * min(std(x), IQR(x)/1.34) * n^{-1/5}     (Silverman 1986, 3.31)
        sigma = std(x)
    scipy.stats.gaussian_kde multiplies this factor by std internally, so
    we return the dimensionless ratio.

    The IQR/1.34 term equals std for Gaussian data (kurtosis 3) but is
    smaller for heavy-tailed data; using min(std, IQR/1.34) makes the
    bandwidth track the bulk width regardless of tail behavior. Without
    this, Scott/Silverman bandwidth on heavy-tailed data oversmooths by
    a factor proportional to sqrt(kurtosis/3): on banana's W2 coords
    (kurt ~40, std/IQR_norm ratio ~2.4), the legacy Scott bandwidth was
    2.4x too large, putting too much mass in the bulk and too little in
    the tails. The result was a log_pdf that violated Gibbs' inequality
    when integrated against samples from the same distribution.
    """
    n = len(x)
    if n < 2:
        return 1.0
    std = float(np.std(x, ddof=1))
    if std <= 0:
        return 1.0
    q25, q75 = np.percentile(x, [25, 75])
    iqr_scale = float((q75 - q25) / 1.34) if (q75 > q25) else std
    sigma_robust = min(std, iqr_scale)
    h = 0.9 * sigma_robust * (n ** (-0.2))
    return h / std


class ScalarMarginal:
    """Wraps empirical CDF (for rank transform) and KDE density (for Jacobian).

    Bandwidth selection: Silverman's robust rule of thumb
    h = 0.9 * min(std, IQR/1.34) * N^{-1/5}, via _silverman_robust_factor.
    This adapts to the *bulk* scale of the data rather than the (possibly
    heavy-tailed) std, avoiding the oversmoothing that scipy's default
    Scott bandwidth produces on high-kurtosis distributions. See the
    _silverman_robust_factor docstring for the failure mode it fixes.
    """

    def __init__(self, x_train):
        x = np.asarray(x_train, dtype=float).ravel()
        x = x[np.isfinite(x)]
        # Deduplicate: PchipInterpolator needs strictly increasing x
        x_sorted = np.sort(x)
        # Add jitter to break exact ties (preserves monotonicity)
        diffs = np.diff(x_sorted)
        if np.any(diffs == 0):
            # Tiny deterministic jitter based on rank
            jitter = np.arange(len(x_sorted)) * 1e-10 * (np.abs(x_sorted).max() + 1)
            x_sorted = x_sorted + jitter
            # Re-sort just in case jitter disrupted ordering
            x_sorted = np.sort(x_sorted)
        self.x_sorted = x_sorted
        self.n = len(x)
        # KDE for density. Robust Silverman bandwidth (IQR-based) avoids
        # heavy-tail oversmoothing; see _silverman_robust_factor.
        self.kde = stats.gaussian_kde(x, bw_method=_silverman_robust_factor(x))
        # Store the input data so __setstate__ can rebuild the KDE (which
        # is not picklable directly: scipy wraps the bw_method in a lambda
        # that pickle can't serialize).
        self._x_train = x
        # For inverse CDF we'll use interpolation
        self._build_cdf_interpolators()
        # Precompute a log-pdf table once at construction so later evaluations
        # are O(log N_grid) per query instead of O(N_train).  This makes batch
        # evaluate_log_density O(N_test) rather than O(N_test * N_train).
        self._build_logpdf_interpolator()

    def _build_logpdf_interpolator(self):
        # Pad ~6 sigma beyond data range to avoid extrapolation in evaluation
        span = self._x_max - self._x_min
        data_std = np.std(self.x_sorted)
        pad = max(6.0 * data_std, 6.0 * self.kde.factor * data_std, 1e-3 * max(span, 1.0))
        grid = np.linspace(self._x_min - pad, self._x_max + pad, 5000)
        # Evaluate KDE once on the grid (this is the expensive step, done only at
        # fit time) and log-transform
        logp_grid = np.log(np.maximum(self.kde(grid), 1e-300))
        self._logpdf_spline = PchipInterpolator(grid, logp_grid, extrapolate=False)
        self._logpdf_left = float(logp_grid[0])
        self._logpdf_right = float(logp_grid[-1])
        self._logpdf_xmin_pad = float(grid[0])
        self._logpdf_xmax_pad = float(grid[-1])

    def _build_cdf_interpolators(self):
        # Augment with tail margin to avoid endpoint degeneracy
        n = self.n
        # Empirical CDF at sorted points, rescaled (k/(n+1) convention)
        F = np.arange(1, n + 1) / (n + 1.0)
        # Monotone interpolation (PCHIP respects monotonicity)
        self._F_forward = PchipInterpolator(self.x_sorted, F, extrapolate=False)
        self._F_inverse = PchipInterpolator(F, self.x_sorted, extrapolate=False)
        self._F_min, self._F_max = F[0], F[-1]
        self._x_min, self._x_max = self.x_sorted[0], self.x_sorted[-1]

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        below = x < self._x_min
        above = x > self._x_max
        mid = ~(below | above)
        out[below] = 0.5 / (self.n + 1.0)  # tail regularization
        out[above] = 1.0 - 0.5 / (self.n + 1.0)
        out[mid] = self._F_forward(x[mid])
        return np.clip(out, 1.0 / (2 * (self.n + 1.0)),
                       1.0 - 1.0 / (2 * (self.n + 1.0)))

    def inverse_cdf(self, u):
        u = np.clip(np.asarray(u, dtype=float),
                    self._F_min + 1e-12, self._F_max - 1e-12)
        return self._F_inverse(u)

    def pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self.kde(x[None, :] if x.ndim == 1 else x).reshape(x.shape)

    def log_pdf(self, x):
        """log of the Gaussian-kernel density estimate.

        Inside the tabulated range [x_min - pad, x_max + pad] this is the
        PCHIP interpolant of log kde on a 5000-point grid; outside it the
        KDE is evaluated directly, so the tails are the Gaussian-mixture
        tails of the estimator and the density integrates to one.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x, dtype=float)
        below = x < self._logpdf_xmin_pad
        above = x > self._logpdf_xmax_pad
        mid = ~(below | above)
        outside = below | above
        if np.any(outside):
            out[outside] = self.kde.logpdf(x[outside])
        out[mid] = self._logpdf_spline(x[mid])
        return out

    def __getstate__(self):
        """Pickle by stripping the unpicklable scipy KDE; keep raw data.

        scipy.stats.gaussian_kde stores its bandwidth as a lambda after
        set_bandwidth() is called (even when given a float arg), and
        lambdas don't pickle. We strip the kde here and rebuild on load.
        """
        state = self.__dict__.copy()
        state.pop("kde", None)
        return state

    def __setstate__(self, state):
        """Rebuild the scipy gaussian_kde after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, "_x_train"):
            self._x_train = self.x_sorted
        self.kde = stats.gaussian_kde(
            self._x_train,
            bw_method=_silverman_robust_factor(self._x_train),
        )


# ---------------------------------------------------------------
# Rank-transform a batch of X samples through stored marginals
# ---------------------------------------------------------------

def rank_transform_through_marginals(X, marginals):
    """
    X: (N, d).  marginals: list of d ScalarMarginal objects.
    Returns Z: (N, d) with Z_i = Phi^{-1}(hat_F_i(X_i)).
    """
    N, d = X.shape
    assert len(marginals) == d
    Z = np.empty_like(X, dtype=float)
    for i in range(d):
        u_i = marginals[i].cdf(X[:, i])
        Z[:, i] = norm.ppf(u_i)
    return Z


# ---------------------------------------------------------------
# Hermite-polynomial evaluation at a batch of points, for a given dictionary
# ---------------------------------------------------------------

def eval_g_polynomial(U, theta, A, K):
    """
    Evaluate tilde_g(u) = sum_{beta in A} theta_beta H_beta(u) at batch U.

    U: (N, r) points.  theta: coefficients indexed by A.  A: list of multi-indices
    (tuples of length r).  K: max total degree.
    Returns (N,) array.
    """
    # Use the slim evaluator: gradients/Hessians not needed for value-only evaluation.
    # At large N (e.g. M_norm QMC points) this saves ~2*d times the memory.
    from .hermite_score_matching import evaluate_features_psi_only
    Psi = evaluate_features_psi_only(U, A, K)
    return Psi @ theta


# ---------------------------------------------------------------
# Mass of a composed estimate on R^d
# ---------------------------------------------------------------

def estimate_log_mass(marginals, log_copula_factor, seed, M=None):
    """Log of the mass on R^d of exp(log_copula_factor(x)) prod_i f_i(x_i), by
    scrambled-Sobol quasi-Monte Carlo against the product of the kernel
    marginals: each coordinate takes a kernel centre drawn uniformly from its
    training sample plus Gaussian jitter of the bandwidth, from one Sobol point
    of dimension 2d.
    """
    from scipy.stats import qmc
    from .config import M_MASS
    d = len(marginals)
    M = int(M_MASS if M is None else M)
    Mp = 1 << int(np.ceil(np.log2(max(M, 2))))
    U01 = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(Mp)
    U01 = np.clip(U01, 1e-12, 1 - 1e-12)
    X = np.empty((Mp, d))
    for i, m in enumerate(marginals):
        centres = m._x_train
        idx = np.minimum((U01[:, 2 * i] * len(centres)).astype(int),
                         len(centres) - 1)
        bw = float(np.sqrt(m.kde.covariance[0, 0]))
        X[:, i] = centres[idx] + bw * norm.ppf(U01[:, 2 * i + 1])
    lf = np.asarray(log_copula_factor(X), dtype=float)
    n_nan = int(np.isnan(lf).sum())
    n_posinf = int(np.isposinf(lf).sum())
    if n_nan or n_posinf:
        raise ValueError(f"estimate_log_mass: {n_nan} NaN and {n_posinf} +inf "
                         f"log-factor values of {Mp}; the mass is not defined")
    # -inf values contribute exp(-inf) = 0 to the mean and are kept as such.
    mx = float(lf[np.isfinite(lf)].max())
    return mx + float(np.log(np.mean(np.exp(lf - mx))))


# ---------------------------------------------------------------
# Reduced-density fitting
# ---------------------------------------------------------------


    def __getstate__(self):
        """Pickle by stripping the unpicklable scipy KDE; keep raw data.
        __setstate__ rebuilds the KDE."""
        state = self.__dict__.copy()
        state.pop("kde", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "_x_train"):
            # Older pickle without _x_train; reconstruct from x_sorted.
            self._x_train = self.x_sorted
        from scipy import stats
        self.kde = stats.gaussian_kde(
            self._x_train, bw_method=_silverman_robust_factor(self._x_train)
        )


class ReducedDensityModel:
    """
    Fits and evaluates the CAS-factorized density approximation on X samples.

    Pipeline:
      1. Fit scalar marginals hat_F_i, hat_f_i for i=1..d
      2. Rank-transform to Z
      3. CAS fit at dim d -> V_r
      4. Project U = Z @ V_r (N x r)
      5. Fit secondary scalar marginals hat_F^U_i, hat_f^U_i on U columns
      6. Re-rank-Gaussianize U -> tilde_U (exactly gamma_1 marginals)
      7. CAS fit at dim r on tilde_U to get tilde_theta parameterizing tilde_g
      8. Monte Carlo normalization constant hat_Z_norm = E_{gamma_r}[exp tilde_g]

    Then evaluate_log_density(X_test) returns log hat_pi_X(X_test).
    """

    def __init__(self, r, K=3, q=3, K_inner=5, q_inner=None,
                 even_degree=False,
                 lambda0=None, M_norm=20000, seed=0,
                 use_secondary_rank=True,
                 weighted_fisher=False, rho=0.0, eps_w=1e-2,
                 lambda_override=None,
                 objective='direct',
                 inner_ridge_scheme='degree_weighted',
                 lambda0_inner=None):
        """
        Parameters
        ----------
        use_secondary_rank : bool
            If True (legacy default), fit the inner polynomial on the
            secondarily rank-Gaussianized coordinates tilde_U and include the
            Jacobian term sum_i log(hat_f^U_i(u_i) / phi(tilde_u_i)) in the
            deployed log-density.

            If False (matches paper §3.2 "deployed likelihood" eq. 3.18), fit
            the inner polynomial L_r directly on U = V_r^T Z and skip the
            secondary rank transform on both fit and eval. The deployed
            log-density is then
                log hat_pi(x) = L_r(u; beta) + sum_i log hat_pi_i(x_i)
                                - log hat_Z_r,
            with hat_Z_r = E_{gamma_r}[exp L_r(.; beta)].

        weighted_fisher : bool
            If True, replace the Stage-2 unweighted Hyvarinen solve by the
            weighted Hyvarinen solve from paper §3.2 eq. (3.5),

                w_j = 1 + rho / (hat_lambda_j + eps_w),

            using the Stage-1 active eigenvalues hat_lambda_j (or
            lambda_override if supplied). When True, requires either an
            outer CAS fit producing eigvals, or lambda_override.

        rho, eps_w : float
            Directional-regularizer strength and floor.
              rho = 0 reduces w_j to 1 (so weighted_fisher=True with rho=0
              should reproduce weighted_fisher=False up to FP roundoff).
              eps_w caps the maximum weight at 1 + rho/eps_w.

        lambda_override : (r,) array or None
            Optional override for the active eigenvalues hat_lambda_j. Useful
            when V_r_override is supplied (no Stage-1 spectrum available) or
            for ablations.

        objective : {'direct', 'psr'}
            Stage-2 cross-term construction (paper §3.2 last paragraph).
              'direct': Hyvarinen integration by parts (default; equation
                eq:direct-inner-solve), uses second derivatives of Phi.
              'psr':   projected-score regression (eq:psr-rhs), plugs in
                Stage-1 projected-score labels y_sc^(n) = V_r^T T(z^(n)) into
                the cross-term. Only meaningful when use_secondary_rank=False
                because the score labels are most natural in u-coordinates.
                Requires Stage-1 score field (so V_r_override mode is
                unsupported for PSR — needs cas_diag['T_train']).
            Both share the same Hessian; they differ only in the empirical
            cross-term construction.

        inner_ridge_scheme : {'degree_weighted', 'plain', 'chaos_block', 'chaos_eig', 'full_eig', 'theoretical', 'heldout'}
            How to regularize the inner Hyvarinen / PSR linear system.
              'degree_weighted' (default): solve_ridge with Lambda=lambda0*|alpha|^2.
                  This is the existing population-heuristic ridge.
              'plain': solve_ridge with Lambda=lambda0*I, no degree weighting.
              'chaos_block': per-chaos-block scalar Tikhonov,
                  tau_k = lambda0 * ||A^{(k,k)}||_op.
              'chaos_eig': per-chaos-block eigendecomposition + per-eigenvalue
                  Tikhonov with conditioning cap kappa_cap (=1e3).
              'full_eig': global eigendecomposition + per-eigenvalue Tikhonov,
                  ignores chaos structure but uses inter-block coupling.
              'hermitescaled:c=C,alpha=A,delta=D': adds
                  C * k^A * sigma_hat_k^2 + D * ||A_inner||_op * k^2 to the
                  k-th chaos block. With c=0,alpha=0,delta=kappa this is the
                  pure Sobolev R_{H^2} regularizer.
              'theoretical:c_cov=C,c_sob=D': R_cov + R_sob + centering
                  (paper §3.3, Eq. R-combined). Legacy form
                  'theoretical:c=C,delta=D' still accepted (c↔c_cov,
                  delta↔c_sob).
        """
        self.r = r
        self.K = K
        self.q = q
        self.even_degree = even_degree
        self.K_inner = K_inner
        self.q_inner = q_inner if q_inner is not None else r
        self.lambda0 = lambda0
        self.lambda0_inner = lambda0_inner if lambda0_inner is not None else lambda0
        self.M_norm = M_norm
        self.seed = seed
        self.use_secondary_rank = use_secondary_rank
        self.weighted_fisher = weighted_fisher
        self.rho = float(rho)
        self.eps_w = float(eps_w)
        self.lambda_override = lambda_override
        assert objective in ('direct', 'psr'), f"objective must be 'direct' or 'psr', got {objective}"
        self.objective = objective
        valid_prefixes = ('degree_weighted', 'plain', 'chaos_block', 'chaos_block_blend', 'chaos_block_mom', 'chaos_block_quantile', 'chaos_block_sigmoid', 'chaos_eig', 'full_eig', 'trace_ridge', 'trace_hybrid', 'theoretical', 'heldout', 'hermitescaled')
        scheme_root = inner_ridge_scheme.split(':', 1)[0]
        assert scheme_root in valid_prefixes, \
            f"inner_ridge_scheme root must be one of {valid_prefixes}, got {scheme_root}"
        self.inner_ridge_scheme = inner_ridge_scheme

    def fit(self, X, V_r_override=None):
        """
        Fit the reduced-density model on samples X.

        If V_r_override is provided (d x r orthonormal), it replaces the CAS
        subspace and skips the outer CAS fit; useful for random-subspace and
        oracle-subspace ablations that hold the rest of the pipeline fixed.
        """
        self.d = X.shape[1]
        self.N = X.shape[0]
        rng = np.random.default_rng(self.seed)

        # Step 1: primary scalar marginals
        self.marginals = _build_marginals_cached(X)

        # Step 2: rank-transform
        Z = rank_transform_through_marginals(X, self.marginals)

        # Step 3: subspace selection
        if V_r_override is None:
            # CAS fit at dim d (streaming — memory-safe at large N, |A|)
            cas_result = cas_fit_streaming(X, K=self.K, q=self.q, r=self.r,
                                           lambda0=self.lambda0, seed=self.seed,
                                           even_degree=self.even_degree)
            self.V_r = cas_result['V_r']             # (d, r)
            self.cas_diag = {
                'stein': cas_result['stein_residual'],
                'eigvals_top': cas_result['eigvals_top'],
                'trC_k': cas_result['trC_k'],
                'E_r': cas_result['E_r'],
                # T_train: (d, N) Stage-1 score field on training samples.
                # Needed by the PSR Stage-2 cross-term (paper eq. 3.7).
                'T_train': cas_result['T_train'],
            }
        else:
            assert V_r_override.shape == (self.d, self.r), \
                f"V_r_override must be ({self.d}, {self.r}), got {V_r_override.shape}"
            self.V_r = V_r_override.copy()
            self.cas_diag = None

        # Step 4: project
        U = Z @ self.V_r                          # (N, r)

        # Steps 5-6: optional secondary rank-Gaussianization U -> tilde_U.
        # The paper §3.2 eq. (3.18) deploys the model directly on U; when
        # use_secondary_rank is False we skip these steps and fit the inner
        # polynomial on U itself. When True (legacy), we keep the secondary
        # rank transform and the deployed-density formula picks up an extra
        # Jacobian factor sum_i log(hat_f^U_i / phi).
        if self.use_secondary_rank:
            self.u_marginals = _build_marginals_cached(U[:, :self.r])
            U_for_fit = rank_transform_through_marginals(U, self.u_marginals)
        else:
            self.u_marginals = None
            U_for_fit = U

        # Step 7: Stage-2 score-matching solve at dim r on the chosen
        # U-coordinates (streaming). Two design knobs:
        #   - weighted_fisher: w_j = 1 + rho/(lambda_j + eps_w), default rho=0
        #   - objective: 'direct' (Hyvarinen integration by parts; eq:direct-inner-solve)
        #               or 'psr' (projected-score regression; eq:psr-rhs).
        from .hermite_score_matching import (build_A_b_streaming, build_weighted_A_b_streaming,
                         build_psr_A_b_streaming, solve_ridge)
        A_inner = enumerate_dictionary(self.r, self.K_inner, self.q_inner)
        self.A_inner = A_inner

        # Resolve weights (used by both objectives).
        if self.weighted_fisher:
            if self.lambda_override is not None:
                lam = np.asarray(self.lambda_override, dtype=float).ravel()
            elif self.cas_diag is not None and 'eigvals_top' in self.cas_diag:
                lam = np.asarray(self.cas_diag['eigvals_top'], dtype=float).ravel()
            else:
                raise ValueError(
                    "weighted_fisher=True requires Stage-1 eigvals or lambda_override; "
                    "neither is available."
                )
            assert lam.shape == (self.r,), f"lambda must be ({self.r},), got {lam.shape}"
            assert np.all(lam >= 0), f"non-negative eigenvalues required, got {lam}"
            w_vec = 1.0 + self.rho / (lam + self.eps_w)
            self.weights = w_vec
            self.lambda_used = lam
        else:
            w_vec = np.ones(self.r, dtype=float)
            self.weights = None
            self.lambda_used = None

        # Resolve objective and build (A, b).
        if self.objective == 'psr':
            # Need Stage-1 score labels in the Stage-2 coordinate system.
            # T_train is (d, N) in z-coordinates. Project to u: (N, r) = T_train.T @ V_r.
            # Then if use_secondary_rank, push score labels through chain rule:
            #     d/d(tilde_u_j) = (1 / (d tilde_u_j / d u_j)) * d/d(u_j)
            # since tilde_u_j = Phi^{-1}(F^U_j(u_j)), Jacobian is hat_f^U_j(u_j)/phi(tilde_u_j).
            if self.cas_diag is None or 'T_train' not in self.cas_diag:
                raise ValueError(
                    "objective='psr' requires Stage-1 fit (no V_r_override) so that "
                    "T_train (Stage-1 score field) is available."
                )
            T_train = self.cas_diag['T_train']        # (d, N) in z-coords
            assert T_train.shape == (self.d, self.N), \
                f"T_train must be ({self.d}, {self.N}), got {T_train.shape}"
            Y_sc_u = (T_train.T) @ self.V_r           # (N, r) in u-coords
            if self.use_secondary_rank:
                # Push to tilde_u via chain rule:
                #   y_sc_tilde_j = y_sc_u_j * (du_j / d tilde_u_j)
                #               = y_sc_u_j * (phi(tilde_u_j) / hat_f^U_j(u_j))
                # Equivalently the score in tilde_u space is the score in u space
                # divided by the forward Jacobian d(tilde_u)/du = hat_f^U_j/phi.
                tilde_U_for_fit = U_for_fit  # already tilde_U in this branch
                Y_sc = np.zeros_like(Y_sc_u)
                for j in range(self.r):
                    log_phi_tilde = norm.logpdf(tilde_U_for_fit[:, j])
                    log_fU = self.u_marginals[j].log_pdf(U[:, j])
                    # forward Jacobian (du -> dtilde_u): hat_f^U_j / phi(tilde_u_j)
                    # backward (we need): phi / hat_f^U_j
                    log_jac_back = log_phi_tilde - log_fU
                    Y_sc[:, j] = Y_sc_u[:, j] * np.exp(log_jac_back)
            else:
                Y_sc = Y_sc_u

            Amat_inner, bvec_inner, _, _, _ = build_psr_A_b_streaming(
                U_for_fit, A_inner, self.K_inner, w_vec, Y_sc, keep_Phi=False)
        elif self.weighted_fisher:
            # Direct Hyvarinen with weights.
            Amat_inner, bvec_inner, _, _, _ = build_weighted_A_b_streaming(
                U_for_fit, A_inner, self.K_inner, w_vec, keep_Phi=False)
        else:
            # Direct Hyvarinen, unweighted (legacy).
            Amat_inner, bvec_inner, _, _, _ = build_A_b_streaming(
                U_for_fit, A_inner, self.K_inner, keep_Phi=False)

        # Inner ridge: choose scheme.
        from .hermite_score_matching import (solve_ridge_chaos_block, solve_ridge_chaos_block_mom,
                         solve_ridge_chaos_block_blend,
                         solve_ridge_chaos_eig, solve_ridge_full)
        if self.inner_ridge_scheme == 'degree_weighted':
            theta_inner, lambda_used = solve_ridge(
                Amat_inner, bvec_inner, A_inner,
                lambda0=self.lambda0_inner, degree_weighted=True)
            ridge_diag = {'lambda0_used': lambda_used}
        elif self.inner_ridge_scheme == 'plain':
            theta_inner, lambda_used = solve_ridge(
                Amat_inner, bvec_inner, A_inner,
                lambda0=self.lambda0_inner, degree_weighted=False)
            ridge_diag = {'lambda0_used': lambda_used}
        elif self.inner_ridge_scheme == 'chaos_block':
            theta_inner, taus = solve_ridge_chaos_block(
                Amat_inner, bvec_inner, A_inner, lambda0=self.lambda0_inner)
            lambda_used = taus
            ridge_diag = {'taus_per_block': taus}
        elif self.inner_ridge_scheme.startswith('chaos_block_sigmoid'):
            from .hermite_score_matching import solve_ridge_chaos_block_sigmoid
            # parse p after colon, e.g. 'chaos_block_sigmoid:2'
            if ':' in self.inner_ridge_scheme:
                p_val = float(self.inner_ridge_scheme.split(':', 1)[1])
            else:
                p_val = 2.0
            theta_inner, sdiag = solve_ridge_chaos_block_sigmoid(
                Amat_inner, bvec_inner, A_inner,
                lambda0=self.lambda0_inner, N=U_for_fit.shape[0], p=p_val,
                return_diag=True)
            lambda_used = sdiag
            ridge_diag = {'sigmoid_diag': sdiag}
        elif self.inner_ridge_scheme.startswith('trace_ridge'):
            # Paper formula (eq:trace-ridge): tau_i^(k) = c*sigma_k^2 + w_i*(lam_max - lam_i)
            # Parse parameters: 'trace_ridge', 'trace_ridge:c=3', 'trace_ridge:c=3,p=4'
            from .hermite_score_matching import solve_ridge_trace
            c_val, p_val = 3.0, 4.0  # paper-deployed defaults
            if ':' in self.inner_ridge_scheme:
                params = self.inner_ridge_scheme.split(':', 1)[1]
                for kv in params.split(','):
                    if '=' in kv:
                        k_, v_ = kv.split('=', 1)
                        if k_.strip() == 'c':
                            c_val = float(v_)
                        elif k_.strip() == 'p':
                            p_val = float(v_)
            theta_inner, tdiag = solve_ridge_trace(
                Amat_inner, bvec_inner, A_inner,
                c=c_val, p=p_val, N=U_for_fit.shape[0],
                return_diag=True)
            lambda_used = tdiag
            ridge_diag = {'trace_diag': tdiag, 'c': c_val, 'p': p_val}
        elif self.inner_ridge_scheme.startswith('trace_hybrid'):
            # Hybrid Λ_tr + Sobolev floor for misspecification:
            # τ_i^(k) = c*σ_hat_k^2 + w_i*(λ_max - λ_i) + δ*||A||_op*k^2
            # Parse: 'trace_hybrid:c=3,p=4,delta=1e-5'
            from .hermite_score_matching import solve_ridge_trace_hybrid
            c_val, p_val, delta_val = 3.0, 4.0, 1e-5  # principled defaults
            if ':' in self.inner_ridge_scheme:
                params = self.inner_ridge_scheme.split(':', 1)[1]
                for kv in params.split(','):
                    if '=' in kv:
                        k_, v_ = kv.split('=', 1)
                        if k_.strip() == 'c': c_val = float(v_)
                        elif k_.strip() == 'p': p_val = float(v_)
                        elif k_.strip() == 'delta': delta_val = float(v_)
            theta_inner, hdiag = solve_ridge_trace_hybrid(
                Amat_inner, bvec_inner, A_inner,
                c=c_val, p=p_val, delta=delta_val, N=U_for_fit.shape[0],
                return_diag=True)
            lambda_used = hdiag
            ridge_diag = {'hybrid_diag': hdiag, 'c': c_val, 'p': p_val, 'delta': delta_val}
        elif self.inner_ridge_scheme.startswith('theoretical'):
            # Paper §3.3 estimator: R_cov + R_sob + centering constraint.
            # Parse: 'theoretical:c_cov=3,c_sob=1e-5' (paper notation) or the
            # legacy form 'theoretical:c=3,delta=1e-5' (still accepted; c↔c_cov,
            # delta↔c_sob).
            from .hermite_score_matching import solve_ridge_theoretical
            c_cov_val, c_sob_val = 3.0, 1e-5
            if ':' in self.inner_ridge_scheme:
                params = self.inner_ridge_scheme.split(':', 1)[1]
                for kv in params.split(','):
                    if '=' in kv:
                        k_, v_ = kv.split('=', 1)
                        key = k_.strip()
                        if key in ('c_cov', 'c'):
                            c_cov_val = float(v_)
                        elif key in ('c_sob', 'delta'):
                            c_sob_val = float(v_)
            theta_inner, tdiag = solve_ridge_theoretical(
                Amat_inner, bvec_inner, A_inner,
                U_train=U_for_fit, c_cov=c_cov_val, c_sob=c_sob_val,
                N=U_for_fit.shape[0], return_diag=True)
            lambda_used = tdiag
            ridge_diag = {'theoretical_diag': tdiag,
                          'c_cov': c_cov_val, 'c_sob': c_sob_val,
                          'c': c_cov_val, 'delta': c_sob_val}
        elif self.inner_ridge_scheme.startswith('heldout'):
            # Held-out selection of the two global scalars (Part I supplement,
            # sec:sm2-selection): split the Stage-2 sample into an estimation
            # part and a held-out part, assemble the score-matching quadratic
            # on each with the same builder as the full-sample quadratic,
            # select (c_cov, c_sob) on a small logarithmic grid by minimizing
            # the held-out objective
            #     J_val(theta) = 0.5 theta' A_val theta - b_val' theta,
            # then refit on the FULL sample at the selected pair.  For fixed
            # coordinates and a candidate fitted without the held-out rows,
            # J_val estimates the reduced Fisher divergence up to a
            # candidate-independent constant.  Here the transform and the
            # subspace are estimated on the full sample before the split, so
            # J_val is used as an empirical tuning criterion.  It needs no
            # reference density, no normalizer, and no noise model.
            # Scheme params: 'heldout:c_cov=3,c_sob=1e-5,frac=0.5,ngrid=5,span=10'
            # where (c_cov, c_sob) are the grid CENTERS and each axis spans
            # [center/span, center*span] with ngrid log-spaced points.
            from .hermite_score_matching import solve_ridge_theoretical
            cov_ctr, sob_ctr = 3.0, 1e-5
            ho_frac, n_grid, grid_span = 0.5, 5, 10.0
            if ':' in self.inner_ridge_scheme:
                params = self.inner_ridge_scheme.split(':', 1)[1]
                for kv in params.split(','):
                    if '=' in kv:
                        k_, v_ = kv.split('=', 1)
                        key = k_.strip()
                        if key in ('c_cov', 'c'):
                            cov_ctr = float(v_)
                        elif key in ('c_sob', 'delta'):
                            sob_ctr = float(v_)
                        elif key == 'frac':
                            ho_frac = float(v_)
                        elif key == 'ngrid':
                            n_grid = int(v_)
                        elif key == 'span':
                            grid_span = float(v_)
            N_fit_all = U_for_fit.shape[0]
            c_cov_sel, c_sob_sel, j_best = cov_ctr, sob_ctr, float('nan')
            if N_fit_all >= 8:
                rng_split = np.random.default_rng(20260715 + int(self.seed or 0))
                perm = rng_split.permutation(N_fit_all)
                n_est = int(round(ho_frac * N_fit_all))
                n_est = min(max(n_est, 4), N_fit_all - 4)
                rows_est, rows_val = perm[:n_est], perm[n_est:]

                def _quad_on(rows):
                    # w_vec is a per-COORDINATE weight vector of length r
                    # (see its construction above); it is NOT per-sample and
                    # passes through the split unchanged.  Only the sample
                    # arrays U_for_fit and Y_sc are sliced by rows.
                    Uf = U_for_fit[rows]
                    if self.objective == 'psr':
                        Am, bv, _, _, _ = build_psr_A_b_streaming(
                            Uf, A_inner, self.K_inner, w_vec, Y_sc[rows],
                            keep_Phi=False)
                    elif self.weighted_fisher:
                        Am, bv, _, _, _ = build_weighted_A_b_streaming(
                            Uf, A_inner, self.K_inner, w_vec, keep_Phi=False)
                    else:
                        Am, bv, _, _, _ = build_A_b_streaming(
                            Uf, A_inner, self.K_inner, keep_Phi=False)
                    return Am, bv

                A_est, b_est = _quad_on(rows_est)
                A_val, b_val = _quad_on(rows_val)
                cov_grid = np.geomspace(cov_ctr / grid_span,
                                        cov_ctr * grid_span, n_grid)
                sob_grid = np.geomspace(sob_ctr / grid_span,
                                        sob_ctr * grid_span, n_grid)
                j_best = float('inf')
                for cc in cov_grid:
                    for cs in sob_grid:
                        th_cand, _ = solve_ridge_theoretical(
                            A_est, b_est, A_inner,
                            U_train=U_for_fit[rows_est],
                            c_cov=float(cc), c_sob=float(cs),
                            N=len(rows_est))
                        j_cand = float(0.5 * th_cand @ A_val @ th_cand
                                       - b_val @ th_cand)
                        if j_cand < j_best:
                            c_cov_sel, c_sob_sel = float(cc), float(cs)
                            j_best = j_cand
            theta_inner, tdiag = solve_ridge_theoretical(
                Amat_inner, bvec_inner, A_inner,
                U_train=U_for_fit, c_cov=c_cov_sel, c_sob=c_sob_sel,
                N=N_fit_all, return_diag=True)
            self.selected_ridge_constants = (c_cov_sel, c_sob_sel)
            self.heldout_selection_diag = {
                'heldout_frac': ho_frac, 'grid_n': n_grid,
                'grid_span': grid_span, 'J_val_min': j_best}
            lambda_used = tdiag
            ridge_diag = {'theoretical_diag': tdiag,
                          'scheme': 'heldout',
                          'heldout_frac': ho_frac, 'grid_n': n_grid,
                          'grid_span': grid_span, 'J_val_min': j_best,
                          'c_cov': c_cov_sel, 'c_sob': c_sob_sel,
                          'c': c_cov_sel, 'delta': c_sob_sel}
        elif self.inner_ridge_scheme.startswith('hermitescaled'):
            # Hermite-scaled BISC (no centering): tau_k = c*k^alpha*sigma_k^2 + delta*||A||_op*k^2
            # Parse: 'hermitescaled:c=0,alpha=0,delta=1e-5'  (used for R_H2 oracle column)
            from .hermite_score_matching import solve_ridge_hermitescaled
            c_val, alpha_val, delta_val = 3.0, 0.0, 1e-5
            if ':' in self.inner_ridge_scheme:
                params = self.inner_ridge_scheme.split(':', 1)[1]
                for kv in params.split(','):
                    if '=' in kv:
                        k_, v_ = kv.split('=', 1)
                        if k_.strip() == 'c': c_val = float(v_)
                        elif k_.strip() == 'alpha': alpha_val = float(v_)
                        elif k_.strip() == 'delta': delta_val = float(v_)
            theta_inner, hsdiag = solve_ridge_hermitescaled(
                Amat_inner, bvec_inner, A_inner,
                c=c_val, alpha=alpha_val, delta=delta_val,
                N=U_for_fit.shape[0], return_diag=True)
            lambda_used = hsdiag
            ridge_diag = {'hermitescaled_diag': hsdiag}
        elif self.inner_ridge_scheme == 'chaos_block_blend':
            theta_inner, blend_diag = solve_ridge_chaos_block_blend(
                Amat_inner, bvec_inner, A_inner,
                lambda0=self.lambda0_inner, N=U_for_fit.shape[0], alpha=4.0)
            lambda_used = blend_diag
            ridge_diag = {'blend_diag': blend_diag}
        elif self.inner_ridge_scheme.startswith('chaos_block_quantile'):
            from .hermite_score_matching import solve_ridge_chaos_block_quantile
            # Parse quantile after colon, e.g. 'chaos_block_quantile:0.5'
            if ':' in self.inner_ridge_scheme:
                q = float(self.inner_ridge_scheme.split(':', 1)[1])
            else:
                q = 0.5
            theta_inner, qdiag = solve_ridge_chaos_block_quantile(
                Amat_inner, bvec_inner, A_inner,
                lambda0=self.lambda0_inner, quantile=q)
            lambda_used = qdiag
            ridge_diag = {'quantile_diag': qdiag}
        elif self.inner_ridge_scheme == 'chaos_block_mom':
            # Median-of-means: rebuild A on per-sample subsets to get robust
            # operator-norm estimates per chaos block.
            theta_inner, taus, rhos = solve_ridge_chaos_block_mom(
                Amat_inner, bvec_inner, A_inner,
                U_train=U_for_fit, K_inner=self.K_inner,
                lambda0=self.lambda0_inner, n_blocks=24)
            lambda_used = taus
            ridge_diag = {'taus_per_block': taus, 'rhos_per_block': rhos}
        elif self.inner_ridge_scheme.startswith('chaos_eig'):
            # Parse: 'chaos_eig' (default mode='inverse'), 'chaos_eig:whiten', 'chaos_eig:proportional'
            parts = self.inner_ridge_scheme.split(':', 1)
            mode = parts[1] if len(parts) > 1 else 'inverse'
            theta_inner, eig_diag = solve_ridge_chaos_eig(
                Amat_inner, bvec_inner, A_inner, lambda0=self.lambda0_inner, mode=mode)
            lambda_used = eig_diag
            ridge_diag = {'chaos_eig_diag': eig_diag, 'mode': mode}
        elif self.inner_ridge_scheme.startswith('full_eig'):
            parts = self.inner_ridge_scheme.split(':', 1)
            mode = parts[1] if len(parts) > 1 else 'inverse'
            theta_inner, eig_diag = solve_ridge_full(
                Amat_inner, bvec_inner, A_inner, lambda0=self.lambda0_inner, mode=mode)
            lambda_used = eig_diag
            ridge_diag = {'full_eig_diag': eig_diag, 'mode': mode}
        else:
            raise ValueError(f"unknown inner_ridge_scheme: {self.inner_ridge_scheme}")

        self.theta_inner = theta_inner
        self.inner_diag = {
            'dict_size': len(A_inner),
            'lambda0_used': lambda_used,
            'theta_norm': float(np.linalg.norm(theta_inner)),
            'ridge_diag': ridge_diag,
            'inner_ridge_scheme': self.inner_ridge_scheme,
        }

        # Step 7b: sup-constrained refinement (the estimator of record when
        # enabled; default from config.CONSTRAINED_DEFAULT, override by
        # setting the attribute constrained_solve on the model). Solves the
        # SAME quadratic over the class {sup_u L <= tau}: the estimate is a
        # probability density by construction, and it equals the
        # unconstrained solution wherever the bound is slack. Only applies
        # to the theoretical/heldout schemes, whose regularizer the
        # constrained assembly mirrors; ablation schemes keep their
        # unconstrained solves by design.
        from .config import CONSTRAINED_DEFAULT
        _use_con = getattr(self, 'constrained_solve', None)
        self.constrained = bool(CONSTRAINED_DEFAULT if _use_con is None
                                else _use_con)
        _scheme_ok = (self.inner_ridge_scheme.startswith('heldout')
                      or self.inner_ridge_scheme.startswith('theoretical'))
        if self.constrained and not _scheme_ok:
            self.constrained = False
        self.constrained_diag = None
        if self.constrained:
            from .constrained import assemble_system, fit_constrained
            tg_unc = eval_g_polynomial(U_for_fit, theta_inner, A_inner,
                                       self.K_inner)
            tau_con = float(np.percentile(tg_unc, 99.9))
            rmax_con = max(18.0, 1.5 * float(
                np.linalg.norm(U_for_fit, axis=1).max()))
            cc_cs = getattr(self, 'selected_ridge_constants', None)
            if cc_cs is None:
                cc_cs = (float(ridge_diag.get('c_cov', 3.0)),
                         float(ridge_diag.get('c_sob', 1e-5)))
            Mq, bq, Cc = assemble_system(
                Amat_inner, bvec_inner, A_inner, U_for_fit,
                U_for_fit.shape[0], cc_cs[0], cc_cs[1], self.K_inner)
            theta_inner, cdiag = fit_constrained(
                Mq, bq, Cc, A_inner, self.K_inner, tau_con, theta_inner,
                rmax_con, U_for_fit.shape[1])
            cdiag['c_cov'], cdiag['c_sob'] = float(cc_cs[0]), float(cc_cs[1])
            self.theta_inner = theta_inner
            self.constrained_diag = cdiag
            self.inner_diag['constrained'] = cdiag

        # Step 8: QMC normalization constant.
        # In both pipelines the integral is E_{gamma_r}[exp tilde_g], because
        # under "secondary rank" the fit coordinate is tilde_U ~ gamma_r by
        # construction, while under the §3.2 direct pipeline the deployed
        # normalizer is E_{gamma_r}[exp L_r] with L_r evaluated against the
        # standard normal reference. The Sobol-QMC code path is identical;
        # the difference is what tg_train (the cap calibrator) is computed on.
        from scipy.stats import qmc

        tg_train = eval_g_polynomial(U_for_fit, theta_inner, A_inner, self.K_inner)
        tg_cap = float(np.percentile(tg_train, 99.9))
        self.tg_cap = tg_cap

        # Sobol sequence in [0,1]^r, scrambled with the same seed for reproducibility
        M = self.M_norm
        # Use next power of 2 >= M for the best Sobol balance
        M_pow2 = 1 << int(np.ceil(np.log2(max(M, 2))))
        sampler = qmc.Sobol(d=self.r, scramble=True, seed=self.seed)
        U01 = sampler.random(M_pow2)  # (M_pow2, r) in [0,1]
        # Clip away from {0, 1} to avoid +/- infinity under inverse normal CDF
        U01 = np.clip(U01, 1e-12, 1 - 1e-12)
        Xi = norm.ppf(U01)  # gamma_r samples via inverse CDF transform

        tg_vals = eval_g_polynomial(Xi, theta_inner, A_inner, self.K_inner)
        if self.constrained:
            # Bounded above by construction: integrate the potential as is.
            tg_clipped = tg_vals
        else:
            tg_clipped = np.minimum(tg_vals, tg_cap)
        max_tg = float(np.max(tg_clipped))
        exp_vals = np.exp(tg_clipped - max_tg)
        self.log_Z_norm = max_tg + float(np.log(np.mean(exp_vals)))
        self.Z_norm_se = float(np.std(exp_vals) / np.sqrt(M_pow2) / np.mean(exp_vals))
        self.Z_norm_clipfrac = float(np.mean(tg_vals > tg_cap))
        self.Z_norm_M_used = int(M_pow2)

        # Step 9: log-mass of the composed estimate (estimate_log_mass).
        self.log_mass = 0.0
        self.log_mass = estimate_log_mass(self.marginals, self._log_copula_factor,
                                          seed=self.seed)

        return self

    def evaluate_log_density(self, X_test):
        """
        Returns log hat_pi_X(X_test): (N_test,) array.

        Two modes (selected at construction time by use_secondary_rank):

        Legacy (use_secondary_rank=True):
            The model fits tilde_g on the secondarily rank-Gaussianized
            coordinates tilde_u = Phi^{-1}(hat_F^U(u)). The induced density
            on U has Lebesgue form
                p_U(u) = exp(tilde_g(tilde_u))/hat_Z * prod_j hat_f^U_j(u_j),
            because the change-of-variables Jacobian d(tilde_u)/du =
            hat_f^U_j/phi(tilde_u_j) cancels the phi(tilde_u_j) inside
            gamma_r(tilde_u). The density of U against gamma_r is then
                tilde_c(u) = p_U(u) / prod_j phi(u_j),
            giving deployed log-density
                log hat_pi_X(x) = tilde_g(tilde_u) + sum_i log hat_f_i(x_i)
                                  + sum_j [log hat_f^U_j(u_j) - log phi(u_j)]
                                  - log hat_Z_norm.

        Paper §3.2 deployed (use_secondary_rank=False, eq. 3.18):
            log hat_pi_X(x) = L_r(u; beta) + sum_i log hat_pi_i(x_i)
                              - log hat_Z_r.
        """
        X_test = np.asarray(X_test, dtype=float)
        N_test = X_test.shape[0]

        # Term: sum_i log hat_f_i(x_i)
        log_marg = np.zeros(N_test)
        for i in range(self.d):
            log_marg += self.marginals[i].log_pdf(X_test[:, i])

        return self._log_copula_factor(X_test) + log_marg - getattr(self, 'log_mass', 0.0)

    def _log_copula_factor(self, X_test):
        """log hat_pi_X(x) - sum_i log hat_f_i(x_i), before mass normalization:
        the fitted reduced factor evaluated through the rank transform (plus
        the secondary-rank Jacobian in the legacy pipeline), minus log Z_r."""
        X_test = np.asarray(X_test, dtype=float)
        N_test = X_test.shape[0]

        # Rank-transform X_test through primary marginals
        Z_test = rank_transform_through_marginals(X_test, self.marginals)
        U_test = Z_test @ self.V_r                # (N_test, r)

        if self.use_secondary_rank:
            # Secondary rank transform: u -> tilde_u (only used for evaluating
            # tilde_g; the Jacobian of tilde_u -> u itself collapses to
            # log hat_f^U_j(u_j) - log phi(u_j) once gamma_r(tilde_u) cancels.)
            tilde_U_test = rank_transform_through_marginals(U_test, self.u_marginals)
            U_for_eval = tilde_U_test

            # Term: sum_j [log hat_f^U_j(u_j) - log phi(u_j)]
            log_jac_inner = np.zeros(N_test)
            for i in range(self.r):
                log_fU = self.u_marginals[i].log_pdf(U_test[:, i])
                log_phi_u = norm.logpdf(U_test[:, i])
                log_jac_inner += (log_fU - log_phi_u)
        else:
            U_for_eval = U_test
            log_jac_inner = 0.0

        # Term: tilde_g(tilde_u) under the legacy pipeline, or L_r(u) directly.
        # Apply the same 99.9th-percentile training cap used for the
        # normalizing-constant integral, to prevent pathological polynomial
        # explosions at evaluation points outside the training support from
        # producing implausibly large log-densities. Without this, on some
        # test configurations the fitted polynomial can spike several
        # log-units above anything seen in training, concentrating the
        # normalized posterior on a handful of spurious grid points. Capping
        # symmetrically with Z_norm is the minimal consistent choice: the
        # same function values that contributed to log_Z_norm are allowed at
        # evaluation time.
        tg = eval_g_polynomial(U_for_eval, self.theta_inner,
                               self.A_inner, self.K_inner)
        if not getattr(self, 'constrained', False):
            tg = np.minimum(tg, self.tg_cap)

        return tg + log_jac_inner - self.log_Z_norm

    # -----------------------------------------------------------
    # Mass normalization on R^d
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # U-coordinate density evaluators (added for the Cor 4.4 / Thm 4.4
    # verification suite in src/cas/stage2_oracle.py).
    #
    # These compute the deployed model's density on the projected
    # coordinates u = V_r^T z directly, without going through x. They are
    # the natural inputs to a Stage-2 KL estimator that conditions on the
    # subspace V_r and measures D_KL(pi_U || hat_pi_U).
    #
    # See the docstring of evaluate_log_density(x) for the underlying
    # derivation of hat_pi_X. The U-coordinate density is the log of
    # hat_pi_X with the marginal term sum_i log hat_f_i(x_i) removed:
    #
    #   log hat_pi_U(u) = log hat_pi_X(x) - sum_i log hat_f_i(x_i)
    #
    # Note this is the density of U against *Lebesgue measure*. For the
    # Holley-Stroock LSI lower bound diagnostic we also need the density
    # ratio hat_c_r(u) = hat_pi_U(u) / gamma_r(u), which is
    # evaluate_log_c_r_U(u). Both are exposed.
    # -----------------------------------------------------------

    def evaluate_log_density_U(self, U_test):
        """Return log hat_pi_U(U_test) - log density on R^r against Lebesgue.

        U_test: (N_test, r) array of points in U-coordinate space.

        Two modes, mirroring evaluate_log_density:

        direct (use_secondary_rank=False):
            log hat_pi_U(u) = L_r(u; beta) - log hat_Z_r + log gamma_r(u)
                            = L_r(u; beta) - log hat_Z_r
                              - 0.5 ||u||^2 - 0.5 r log(2 pi)

        secondary-rank (use_secondary_rank=True):
            log hat_pi_U(u) = g_tilde(u_tilde(u)) - log hat_Z_r
                              + sum_j log hat_f^U_j(u_j)
            (the gamma_r(u_tilde) and Jacobian phi(tilde_u_j) factors cancel;
            see derivation in the class docstring).
        """
        U_test = np.asarray(U_test, dtype=float)
        assert U_test.ndim == 2 and U_test.shape[1] == self.r, \
            f"U_test must be (N, {self.r}), got {U_test.shape}"
        N_test = U_test.shape[0]

        if self.use_secondary_rank:
            tilde_U = rank_transform_through_marginals(U_test, self.u_marginals)
            U_for_eval = tilde_U
            # Jacobian: sum_j log hat_f^U_j(u_j). (The gamma_r(tilde_u) and
            # Jacobian phi(tilde_u_j) terms have already cancelled in the
            # derivation; here we only need the f^U_j term.)
            log_jac = np.zeros(N_test)
            for j in range(self.r):
                log_jac += self.u_marginals[j].log_pdf(U_test[:, j])
        else:
            U_for_eval = U_test
            # Direct mode: log gamma_r(u) = -0.5 ||u||^2 - 0.5 r log(2 pi)
            log_jac = (-0.5 * np.sum(U_test ** 2, axis=1)
                       - 0.5 * self.r * np.log(2 * np.pi))

        tg = eval_g_polynomial(U_for_eval, self.theta_inner,
                               self.A_inner, self.K_inner)
        if not getattr(self, 'constrained', False):
            tg = np.minimum(tg, self.tg_cap)

        return tg + log_jac - self.log_Z_norm

    def evaluate_log_c_r_U(self, U_test):
        """Return log hat_c_r(U_test) - log density ratio against gamma_r.

        hat_c_r(u) := hat_pi_U(u) / gamma_r(u), so this is just
        evaluate_log_density_U(u) minus log gamma_r(u).

        Useful for the Holley-Stroock LSI lower bound diagnostic
        (lsi_holley_stroock_lb in cas.stage2_oracle).
        """
        U_test = np.asarray(U_test, dtype=float)
        log_pi_U = self.evaluate_log_density_U(U_test)
        log_gamma_r = (-0.5 * np.sum(U_test ** 2, axis=1)
                       - 0.5 * self.r * np.log(2 * np.pi))
        return log_pi_U - log_gamma_r

    def evaluate_grad_log_density_U(self, U_test):
        """Return grad_u log hat_pi_U(U_test) - gradient wrt u.

        U_test: (N_test, r) array.
        Returns: (N_test, r) array.

        DIRECT MODE ONLY for now. For use_secondary_rank=True the chain rule
        through tilde_u = Phi^{-1}(hat_F^U(u)) requires per-axis derivatives
        of hat_f^U_j (Jacobian) and the score of the inner polynomial in
        tilde_u coordinates. Computable but adds machinery; we don't need it
        for the immediate verification (which fits the model with
        use_secondary_rank=False to isolate Stage-2 error from marginal
        Jacobian effects).

        In direct mode:
            log hat_pi_U(u) = L_r(u; beta) - log hat_Z_r + log gamma_r(u)
            grad_u log hat_pi_U(u) = grad_u L_r(u; beta) - u

        grad_u L_r is computed via the Hermite gradient identity
            d_j H_alpha(u) = sqrt(alpha_j) H_{alpha - e_j}(u)
        which is what evaluate_features returns in its Phi_list.
        """
        if self.use_secondary_rank:
            raise NotImplementedError(
                "evaluate_grad_log_density_U requires use_secondary_rank=False "
                "for the current implementation. The secondary-rank chain rule "
                "is doable but unused by the immediate verification pipeline; "
                "see the docstring."
            )
        U_test = np.asarray(U_test, dtype=float)
        assert U_test.ndim == 2 and U_test.shape[1] == self.r, \
            f"U_test must be (N, {self.r}), got {U_test.shape}"

        # Hermite gradient: evaluate_features returns Phi_list, a list of r
        # arrays each (N, |A_inner|), where Phi_list[j][:, a] =
        #   sqrt(alpha_j) * H_{alpha - e_j}(u).
        # The gradient of L_r is then Phi_list[j] @ theta_inner.
        from .hermite_score_matching import evaluate_features
        _, Phi_list, _ = evaluate_features(U_test, self.A_inner, self.K_inner)
        grad_L = np.column_stack([
            Phi_list[j] @ self.theta_inner for j in range(self.r)
        ])  # (N, r)

        # On the unconstrained (ablation) path the evaluated log-density is
        # min(L, cap), so wherever L exceeds the cap its gradient is zero and
        # the density gradient is -u alone. The constrained path applies no
        # cap and takes grad L everywhere.
        if not getattr(self, 'constrained', False):
            tg = eval_g_polynomial(U_test, self.theta_inner,
                                   self.A_inner, self.K_inner)
            grad_L[tg > self.tg_cap] = 0.0

        # log gamma_r gradient is just -u
        return grad_L - U_test


# ---------------------------------------------------------------
# Baselines for the test-log-likelihood experiment
# ---------------------------------------------------------------

class ProductOfMarginalsModel:
    """Independence baseline: hat_pi_X(x) = prod_i hat_f_i(x_i)."""

    def fit(self, X):
        self.d = X.shape[1]
        self.marginals = _build_marginals_cached(X)
        return self

    def evaluate_log_density(self, X_test):
        N_test = X_test.shape[0]
        log_p = np.zeros(N_test)
        for i in range(self.d):
            log_p += self.marginals[i].log_pdf(X_test[:, i])
        return log_p


class GaussianCopulaModel:
    """
    Semiparametric: scalar KDE marginals + full-rank Gaussian copula.
    log pi_X(x) = log c_Gauss(Phi(z(x))) + sum_i log hat_f_i(x_i) - log A
    with log c_Gauss = -(1/2) z^T (R^{-1} - I) z - (1/2) log det R and A the
    mass of the composed estimate on R^d (estimate_log_mass).
    """

    def fit(self, X):
        self.d = X.shape[1]
        self.marginals = _build_marginals_cached(X)
        # Estimate correlation in rank-Gaussianized coordinates
        Z = rank_transform_through_marginals(X, self.marginals)
        self.R = np.corrcoef(Z.T)
        # Regularize a touch to avoid singularity
        self.R += 1e-6 * np.eye(self.d)
        self.R_inv = np.linalg.inv(self.R)
        sign, logdet = np.linalg.slogdet(self.R)
        self.log_det_R = logdet
        self.log_mass = 0.0
        self.log_mass = estimate_log_mass(self.marginals, self._log_copula_factor,
                                          seed=0)
        return self

    def _log_copula_factor(self, X_test):
        Z = rank_transform_through_marginals(np.asarray(X_test, dtype=float),
                                             self.marginals)
        # log c_Gauss(Phi(z)) = -(1/2) z^T (R^{-1} - I) z - (1/2) log det R
        M = self.R_inv - np.eye(self.d)
        quad = np.einsum('ni,ij,nj->n', Z, M, Z)
        return -0.5 * quad - 0.5 * self.log_det_R

    def evaluate_log_density(self, X_test):
        N_test = X_test.shape[0]
        log_p = np.zeros(N_test)
        for i in range(self.d):
            log_p += self.marginals[i].log_pdf(X_test[:, i])
        return log_p + self._log_copula_factor(X_test) - getattr(self, 'log_mass', 0.0)


class FullDKDEModel:
    """
    Full-d KDE on rank-Gaussianized coordinates, then pull back to X-space:
      hat_f_Z(z) from KDE, hat_pi_X(x) = hat_f_Z(z(x)) * prod_i hat_f_i(x_i)/varphi(z_i)
    Feasible only at moderate d.
    """

    def fit(self, X):
        self.d = X.shape[1]
        self.marginals = _build_marginals_cached(X)
        Z = rank_transform_through_marginals(X, self.marginals)
        self.kde_Z = stats.gaussian_kde(Z.T)
        return self

    def evaluate_log_density(self, X_test):
        Z = rank_transform_through_marginals(X_test, self.marginals)
        N_test = X_test.shape[0]
        log_f_Z = np.log(np.maximum(self.kde_Z(Z.T), 1e-300))
        # Jacobian pullback
        log_jac = np.zeros(N_test)
        for i in range(self.d):
            log_f_i = self.marginals[i].log_pdf(X_test[:, i])
            log_phi = norm.logpdf(Z[:, i])
            log_jac += (log_f_i - log_phi)
        return log_f_Z + log_jac


if __name__ == "__main__":
    # Smoke test on a small Gaussian copula
    rng = np.random.default_rng(0)
    d, N = 10, 1000
    R = np.eye(d)
    R[0, 1] = R[1, 0] = 0.7
    R[2, 3] = R[3, 2] = 0.5
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((N, d)) @ L.T
    # Apply non-Gaussian marginals (exponential via PIT)
    X = np.empty_like(Z)
    for i in range(d):
        u = norm.cdf(Z[:, i])
        X[:, i] = -np.log(1 - u)  # exponential marginal

    Ntr = 800
    X_tr, X_te = X[:Ntr], X[Ntr:]

    print("Fitting ProductOfMarginals...")
    pom = ProductOfMarginalsModel().fit(X_tr)
    print(f"  test log-lik: {pom.evaluate_log_density(X_te).mean():.4f}")

    print("Fitting GaussianCopula...")
    gc = GaussianCopulaModel().fit(X_tr)
    print(f"  test log-lik: {gc.evaluate_log_density(X_te).mean():.4f}")

    print("Fitting FullDKDE...")
    kde = FullDKDEModel().fit(X_tr)
    print(f"  test log-lik: {kde.evaluate_log_density(X_te).mean():.4f}")

    print("Fitting ReducedDensityModel (r=2)...")
    rdm = ReducedDensityModel(r=2, K=2, q=2, K_inner=5).fit(X_tr)
    print(f"  test log-lik: {rdm.evaluate_log_density(X_te).mean():.4f}")
    print(f"  CAS Stein: {rdm.cas_diag['stein']:.4f}")
    print(f"  E_r: {rdm.cas_diag['E_r']:.4f}")
    print(f"  log Z_norm: {rdm.log_Z_norm:.4f}")
