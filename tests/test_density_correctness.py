"""Tests of the analytical noise densities (banana, cubic_banana, ppg).

The fundamental identity tested is **Stein/IBP**: for any smooth density
p(x) on R^d that decays at infinity,

    E_{x ~ p}[(d/dx_i) log p(x) * x_j] = -delta_{ij}                  (1)

equivalently, the Fisher-times-second-moment cross term satisfies

    E_p[grad log p * x^T] = -I.                                       (2)

This is a *theorem*, not an approximation, and must hold to MC accuracy
~ O(sigma / sqrt(N)) where sigma is the empirical std of the integrand.

We also test:
  - the score, computed by FD of log_density, has zero empirical mean
    under p (E_p[grad log p] = 0 by direct Stein)
  - the sampler and density are mutually consistent: a re-evaluation of
    log_density at sampler outputs gives finite values with realistic
    distribution
  - extreme tails are not clipped: at N=1M, max|eta| grows with N (a clipped
    sampler would cap max|eta| at a fixed value regardless of N)

Per-benchmark thresholds for the IBP test reflect the *intrinsic* MC
noise of each benchmark, which depends on the variance of grad log p
under p.  Banana W_2 has fat tails -> larger MC variance -> looser threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from cas.noise import sample_noise, log_noise_density
from cas.cubic_banana import sample_cubic_banana_noise, log_cubic_banana_density
from cas.ppg import sample_ppg_noise, log_ppg_density
from cas.config import D_OBS


N_IBP_TEST = 100_000  # large enough for IBP test, small enough for fast CI
FD_H = 1e-4


def _fd_score(eta: np.ndarray, log_density_fn) -> np.ndarray:
    """Central FD of log_density_fn w.r.t. each coordinate."""
    N, d = eta.shape
    grad = np.empty_like(eta)
    for j in range(d):
        ep = eta.copy(); ep[:, j] += FD_H
        em = eta.copy(); em[:, j] -= FD_H
        grad[:, j] = (log_density_fn(ep) - log_density_fn(em)) / (2.0 * FD_H)
    return grad


class TestBananaIBP:
    """Stein/IBP identities for the banana noise law in eta-space."""

    @pytest.fixture(scope="class")
    def eta_and_grad(self):
        rng = np.random.default_rng(0)
        eta = sample_noise(N_IBP_TEST, rng)
        grad = _fd_score(eta, log_noise_density)
        return eta, grad

    def test_stein_zero_mean(self, eta_and_grad):
        """E_p[grad log p] = 0 (basic Stein identity)."""
        eta, grad = eta_and_grad
        norm = float(np.linalg.norm(grad.mean(axis=0)))
        # Banana's W2 has fat tails -> moderate MC variance on this stat.
        # At N=100K, expect ||E[grad]|| ~ O(0.1).
        assert norm < 0.3, f"||E[grad log p]|| = {norm:.4f} too large"

    def test_ibp_cross_term(self, eta_and_grad):
        """E_p[grad log p * eta^T] = -I, the integration-by-parts identity."""
        eta, grad = eta_and_grad
        d = D_OBS
        ETeta = (grad.T @ eta) / N_IBP_TEST
        residual_F = float(np.linalg.norm(ETeta + np.eye(d)))
        # Monte Carlo noise at N = 1e5 is 0.25-0.35; threshold 0.5.
        assert residual_F < 0.5, (
            f"||E[grad log p * eta^T] + I||_F = {residual_F:.4f}. "
            f"This should be small by integration by parts."
        )

    def test_ibp_diagonal_near_minus_one(self, eta_and_grad):
        """Each diagonal entry of E[grad log p * eta^T] should be ~ -1."""
        eta, grad = eta_and_grad
        ETeta = (grad.T @ eta) / N_IBP_TEST
        diag = np.diag(ETeta)
        # All diagonals should be in [-1.2, -0.8] at this N.
        assert np.all(np.abs(diag + 1.0) < 0.25), (
            f"diagonal not near -1: {diag}"
        )


class TestCubicBananaIBP:
    """Stein/IBP identities for cubic-banana."""

    @pytest.fixture(scope="class")
    def eta_and_grad(self):
        rng = np.random.default_rng(0)
        eta = sample_cubic_banana_noise(N_IBP_TEST, rng)
        grad = _fd_score(eta, log_cubic_banana_density)
        return eta, grad

    def test_stein_zero_mean(self, eta_and_grad):
        eta, grad = eta_and_grad
        norm = float(np.linalg.norm(grad.mean(axis=0)))
        assert norm < 0.3, f"||E[grad log p]|| = {norm:.4f}"

    def test_ibp_cross_term(self, eta_and_grad):
        eta, grad = eta_and_grad
        d = eta.shape[1]
        ETeta = (grad.T @ eta) / N_IBP_TEST
        residual_F = float(np.linalg.norm(ETeta + np.eye(d)))
        # Cubic banana has heavier-than-Gaussian tails on W2 (cubic perturbation).
        # Threshold 0.5 at N = 1e5.
        assert residual_F < 0.5, (
            f"||E[grad log p * eta^T] + I||_F = {residual_F:.4f} on cubic_banana"
        )

    def test_ibp_diagonal_near_minus_one(self, eta_and_grad):
        eta, grad = eta_and_grad
        ETeta = (grad.T @ eta) / N_IBP_TEST
        diag = np.diag(ETeta)
        assert np.all(np.abs(diag + 1.0) < 0.25), f"diagonal: {diag}"


class TestPPGIBP:
    """Stein/IBP identities for PPG."""

    @pytest.fixture(scope="class")
    def eta_and_grad(self):
        rng = np.random.default_rng(0)
        eta = sample_ppg_noise(N_IBP_TEST, rng)
        grad = _fd_score(eta, log_ppg_density)
        return eta, grad

    def test_stein_zero_mean(self, eta_and_grad):
        eta, grad = eta_and_grad
        norm = float(np.linalg.norm(grad.mean(axis=0)))
        assert norm < 0.2, f"||E[grad log p]|| = {norm:.4f}"

    def test_ibp_cross_term(self, eta_and_grad):
        eta, grad = eta_and_grad
        d = eta.shape[1]
        ETeta = (grad.T @ eta) / N_IBP_TEST
        residual_F = float(np.linalg.norm(ETeta + np.eye(d)))
        # PPG has sub-Gaussian tails (quartic potential dampens tails),
        # so MC noise floor is lower.
        assert residual_F < 0.3, (
            f"||E[grad log p * eta^T] + I||_F = {residual_F:.4f} on PPG"
        )

    def test_ibp_diagonal_near_minus_one(self, eta_and_grad):
        eta, grad = eta_and_grad
        ETeta = (grad.T @ eta) / N_IBP_TEST
        diag = np.diag(ETeta)
        assert np.all(np.abs(diag + 1.0) < 0.15), f"diagonal: {diag}"


class TestNoClipSaturation:
    """At N >= 100K on banana, the sampler should produce |eta| values
    well beyond the old clip ceiling of 27.6.  This test directly catches
    a regression to clip-based sampling.
    """

    def test_banana_eta_uncapped(self):
        """The logit-Phi transform is uncapped (no clip at the logit ceiling
        27.63). The bounded-bend (tanh) banana is light-tailed, so
        max|eta| stays below that ceiling while still growing with N; the no-clip
        property is therefore tested directly on the transform."""
        from cas.transforms import logit_phi
        # Direct no-clip check: a z past the old clip maps past the old ceiling.
        assert float(logit_phi(np.array([7.5]))[0]) > 28.0
        # Sampler: finite, light-tailed (bounded bend), and growing with N.
        m_small = float(np.abs(sample_noise(50_000, np.random.default_rng(0))).max())
        m_large = float(np.abs(sample_noise(500_000, np.random.default_rng(1))).max())
        assert np.isfinite(m_large)
        assert m_large > m_small, "max|eta| should grow with N (uncapped)"
        assert m_large < 27.0, "bounded-bend law is light-tailed (below old clip)"

    def test_cubic_banana_eta_uncapped(self):
        """max|eta| on cubic_banana also grows with N."""
        rng = np.random.default_rng(0)
        eta = sample_cubic_banana_noise(500_000, rng)
        max_abs = float(np.abs(eta).max())
        # Cubic banana also has fat-ish tails (W2 ~ cubic of W1).
        assert max_abs > 28.0, (
            f"max|eta| = {max_abs:.2f} at N=500K on cubic_banana. "
            f"Should exceed 28 with the stable transforms."
        )


class TestSamplerDensityConsistency:
    """The sampler and the density must be mutually consistent.

    If `sampler(N, rng)` draws from `exp(log_density)`, then the
    self-normalized importance sampling identity at the sampler's own samples
    is trivially `E_p[1] = 1`.  A meaningful consistency test is:
    distribution of log_density at sampler outputs has a sensible mean
    (the negative differential entropy) and finite variance.
    """

    def test_banana_density_finite_at_samples(self):
        rng = np.random.default_rng(0)
        eta = sample_noise(10_000, rng)
        ld = log_noise_density(eta)
        assert np.all(np.isfinite(ld))
        # Differential entropy of banana noise is some fixed value;
        # mean log density is its negative.  Just check the value is sane
        # (not all -inf or +inf).
        mean_ld = float(ld.mean())
        # In d=20 with banana structure, log density is roughly -d/2 log(2pi)
        # plus corrections; expect mean ~ -30 to -50 nats.
        assert -100 < mean_ld < 0, f"mean log density out of range: {mean_ld}"

    def test_cubic_banana_density_finite_at_samples(self):
        rng = np.random.default_rng(0)
        eta = sample_cubic_banana_noise(10_000, rng)
        ld = log_cubic_banana_density(eta)
        assert np.all(np.isfinite(ld))
        mean_ld = float(ld.mean())
        # d=25, expect mean log density in similar range.
        assert -150 < mean_ld < 0, f"mean log density out of range: {mean_ld}"

    def test_ppg_density_finite_at_samples(self):
        rng = np.random.default_rng(0)
        eta = sample_ppg_noise(10_000, rng)
        ld = log_ppg_density(eta)
        assert np.all(np.isfinite(ld))
        mean_ld = float(ld.mean())
        assert -100 < mean_ld < 0, f"mean log density out of range: {mean_ld}"


@pytest.mark.slow
class TestProductionScale:
    """Slow tests exercising production-scale N (up to 5M).

    Verify the transforms / density / sampler stay numerically sane at the
    sample sizes actually used in paper experiments. Marked @slow; run with
    pytest -m slow.
    """

    def test_banana_sampler_at_N5M_finite(self):
        """sample_noise must produce finite values at N=5M. The bounded-bend (tanh)
        law is light-tailed, so max|eta| is modest (the old heavy quadratic banana
        exceeded 50; tanh stays well under 30)."""
        rng = np.random.default_rng(0)
        eta = sample_noise(5_000_000, rng)
        assert np.all(np.isfinite(eta))
        max_abs = float(np.abs(eta).max())
        assert 10.0 < max_abs < 30.0, (
            f"max|eta| = {max_abs:.2f} at N=5M; expected finite, uncapped, and "
            f"light-tailed for the bounded-bend law."
        )

    def test_banana_density_at_N1M_IBP_tight(self):
        """At N=1M, the IBP residual should be tight (<0.2 F-norm).
        Confirms the fix scales correctly to production N."""
        rng = np.random.default_rng(0)
        eta = sample_noise(1_000_000, rng)
        h = 1e-4
        d = D_OBS
        grad = np.empty_like(eta)
        for j in range(d):
            ep = eta.copy(); ep[:, j] += h
            em = eta.copy(); em[:, j] -= h
            grad[:, j] = (log_noise_density(ep) - log_noise_density(em)) / (2 * h)
        ETeta = (grad.T @ eta) / 1_000_000
        residual_F = float(np.linalg.norm(ETeta + np.eye(d)))
        # Monte Carlo noise at N = 1e6 is about 0.08; threshold 0.20.
        assert residual_F < 0.20, (
            f"||E[grad log p eta^T] + I||_F at N=1M = {residual_F:.4f}; "
            f"expected about 0.08."
        )

    def test_cubic_banana_sampler_at_N5M_finite(self):
        rng = np.random.default_rng(0)
        eta = sample_cubic_banana_noise(5_000_000, rng)
        assert np.all(np.isfinite(eta))

    def test_ppg_sampler_at_N2M_finite(self):
        # PPG uses rejection sampling on a quartic potential; can be slow.
        rng = np.random.default_rng(0)
        eta = sample_ppg_noise(2_000_000, rng)
        assert np.all(np.isfinite(eta))


# ---------------------------------------------------------------------------
# z-space Stein tests via cas.rank_gauss_density.compute_population_MHC
#
# Tests of the path from the eta-space density to the rank-Gaussianized
# z-space density through empirical per-coordinate marginals. The path under test is exactly what the supplement's
# bound-comparison script uses, so a regression here would corrupt that
# computation.
#
# The identity tested is z-space Stein:
#     E_{z ~ pi_Z}[grad log pi_Z(z) * z^T] = -I
# where pi_Z is the rank-Gaussianized density built by make_log_pi_Z.
#
# At N=20K we expect MC noise of order 1/sqrt(N) ~ 0.007 per matrix entry;
# Frobenius residuals ~0.2-0.8 are normal (the variance of the integrand
# is large on heavy-tailed coords).
# ---------------------------------------------------------------------------

N_ZSTEIN_TEST = 20_000  # smaller than the eta-space tests because per-eval
                          # cost is dominated by per-coord KDE log-pdf, not
                          # a closed-form density. At N=20K, the d=25
                          # cubic_banana case takes ~10s end-to-end.
FD_H_Z = 1e-3            # KDE log-pdf is not C^infty; central FD at h=1e-4
                          # picks up bandwidth-scale roughness. h=1e-3 is
                          # comfortably above the KDE smoothing scale and
                          # well below the natural z-scale of 1.


class TestZSpaceStein:
    """Stein identity in the rank-Gauss z-frame for each benchmark.

    Constructs log_pi_Z via cas.make_log_pi_Z with empirical per-coord
    marginals, FDs it for the score, and checks the cross-term identity.
    Thresholds sit above the Monte Carlo noise at N = 2e4.
    """

    @pytest.fixture(scope="class")
    def banana_population(self):
        from cas import compute_population_MHC
        return compute_population_MHC(
            sample_noise, log_noise_density,
            N=N_ZSTEIN_TEST, seed=0, h_fd=FD_H_Z,
        )

    @pytest.fixture(scope="class")
    def cubic_banana_population(self):
        from cas import compute_population_MHC
        return compute_population_MHC(
            sample_cubic_banana_noise, log_cubic_banana_density,
            N=N_ZSTEIN_TEST, seed=0, h_fd=FD_H_Z,
        )

    @pytest.fixture(scope="class")
    def ppg_population(self):
        from cas import compute_population_MHC
        return compute_population_MHC(
            sample_ppg_noise, log_ppg_density,
            N=N_ZSTEIN_TEST, seed=0, h_fd=FD_H_Z,
        )

    def test_banana_ibp_residual(self, banana_population):
        """||E[score z^T] + I||_F small on banana at N=20K.

        Banana W2 has the heaviest tails of the three benchmarks; the typical
        residual is about 0.7 and the threshold 1.0.
        """
        assert banana_population["ibp_residual_F"] < 1.0, (
            f"||E[score z^T] + I||_F = {banana_population['ibp_residual_F']:.3f} "
            f"on banana at N=20K. Should be ~0.7 with the correct log_pi_Z."
        )

    def test_cubic_banana_ibp_residual(self, cubic_banana_population):
        """||E[score z^T] + I||_F small on cubic_banana.

        The rank-Gaussianization here is not inv_logit_phi(eta), because the
        W2 coordinate has a non-logistic marginal. Typical residual about 0.4,
        threshold 0.7.
        """
        assert cubic_banana_population["ibp_residual_F"] < 0.7, (
            f"||E[score z^T] + I||_F = {cubic_banana_population['ibp_residual_F']:.3f} "
            f"on cubic_banana at N=20K. A wrong log_pi_Z (e.g. banana's copula"
            f" density applied to cubic_banana samples) gives ~2.0 here."
        )

    def test_ppg_ibp_residual(self, ppg_population):
        """||E[score z^T] + I||_F small on ppg.

        PPG has sub-Gaussian tails (quartic potential), so the MC noise floor
        is lower than the other two. Threshold 0.4 is above the ~0.17 typical
        noise but catches any density-misspecification regression.
        """
        assert ppg_population["ibp_residual_F"] < 0.4, (
            f"||E[score z^T] + I||_F = {ppg_population['ibp_residual_F']:.3f} "
            f"on ppg at N=20K."
        )

    @pytest.mark.parametrize("benchmark", ["banana", "cubic_banana", "ppg"])
    def test_diagonal_near_minus_one(self, benchmark, request):
        pop = request.getfixturevalue(f"{benchmark}_population")
        ibp = (pop["score"].T @ pop["z"]) / pop["N"]
        diag = np.diag(ibp)
        # All diagonals should be near -1 (the per-coord Stein identity).
        worst = float(np.max(np.abs(diag + 1.0)))
        # At N=20K the per-coord diagonal noise is ~0.1; allow up to 0.3.
        assert worst < 0.3, (
            f"{benchmark}: worst diagonal deviation from -1 is {worst:.3f}. "
            f"diag = {diag.round(3).tolist()}"
        )

    @pytest.mark.parametrize("benchmark", ["banana", "cubic_banana", "ppg"])
    def test_z_marginals_are_standard_normal(self, benchmark, request):
        """Rank-Gaussianized z has per-coord N(0,1) marginals by construction.

        This is a sanity check: if z marginals aren't N(0,1), the per-coord
        rank-Gauss step is broken (e.g. using an analytical inverse for a
        benchmark whose marginals don't match it).
        """
        pop = request.getfixturevalue(f"{benchmark}_population")
        z = pop["z"]
        means = z.mean(axis=0)
        stds = z.std(axis=0)
        assert np.max(np.abs(means)) < 0.05, f"{benchmark}: z means = {means}"
        assert np.max(np.abs(stds - 1.0)) < 0.05, f"{benchmark}: z stds = {stds}"

    @pytest.mark.parametrize("benchmark", ["banana", "cubic_banana", "ppg"])
    def test_bound_dominance_with_C_stein(self, benchmark, request):
        """J_hi(V) <= 0.5 tr((I-VV')C_stein) at V = top-r of C_stein.

        This is the algebraic dominance from LCLMZ that the supplement's
        comparison relies on. With C := H + M - 2I (Stein-implied form),
        the dominance is an algebraic identity (proved by x - 1 - log x >= 0
        on eigvals of H_perp) and must hold at every finite N, modulo
        small numerical tolerance from the slogdet step.

        Note that with C := T T^T / N (direct estimator), the dominance
        holds only up to MC noise on the Stein identity (which can exceed
        the J_hi-vs-J_trace gap at small N). This test uses C_stein for
        an unambiguous check.
        """
        from cas.jhi_cas import J_hi
        pop = request.getfixturevalue(f"{benchmark}_population")
        M = pop["M"]
        H = pop["H"]
        C = pop["C_stein"]
        d = M.shape[0]
        # Pick a representative r per benchmark (matches deployment).
        r_map = {"banana": 4, "cubic_banana": 6, "ppg": 3}
        r = r_map[benchmark]
        # V_pop = top-r of C_stein
        w, U = np.linalg.eigh(C)
        order = np.argsort(-w)
        V_pop = U[:, order[:r]]
        P = np.eye(d) - V_pop @ V_pop.T
        j_trace = 0.5 * float(np.trace(P @ C))
        j_hi = J_hi(V_pop, M, H)
        assert j_hi is not None, f"{benchmark}: J_hi returned None (H_perp not PD)"
        # Algebraic identity; allow 1e-6 tolerance for slogdet roundoff.
        assert j_hi <= j_trace + 1e-6, (
            f"{benchmark}: J_hi={j_hi:.4f} > J_trace={j_trace:.4f} "
            f"(gap={j_trace - j_hi:.4e}). This violates the algebraic dominance"
            f" with C := H + M - 2I; check the J_hi formula."
        )
