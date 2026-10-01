"""Tests of the constrained Stage-2 solve: what its diagnostics assert."""
import numpy as np
import pytest

from cas import constrained as C


def _tiny_system(r=2, K_in=4, seed=0):
    """A small Stage-2 quadratic on a Gaussian sample, assembled the way the
    estimator assembles it, so fit_constrained runs its real code path."""
    from cas.hermite_score_matching import enumerate_dictionary, build_A_b_streaming
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((400, r))
    A_inner = enumerate_dictionary(r, K_in, r)
    Amat, bvec, _, _, _ = build_A_b_streaming(U, A_inner, K_in, keep_Phi=False)
    M, b, Cc = C.assemble_system(Amat, bvec, A_inner, U, U.shape[0], 3.0, 1e-5, K_in)
    beta_unc = np.linalg.solve(M, b)
    return M, b, Cc, A_inner, K_in, beta_unc, U


def test_converged_requires_finite_evaluations(monkeypatch):
    """A search that meets both tolerances but met a nonfinite polynomial
    value inside the ball must not report convergence; the count is kept."""
    M, b, Cc, A_inner, K_in, beta_unc, U = _tiny_system()
    real = C._eval_poly

    def poisoned(P, theta, A_inner_, K_in_):
        out = np.asarray(real(P, theta, A_inner_, K_in_), dtype=float)
        if out.ndim == 1 and out.size > 100:
            out = out.copy()
            out[7] = np.nan          # one nonfinite value on the search grid
        return out

    monkeypatch.setattr(C, "_eval_poly", poisoned)
    beta, diag = C.fit_constrained(M, b, Cc, A_inner, K_in, tau=5.0,
                                   beta_unc=beta_unc, rmax=18.0, r=2)
    assert diag["n_nonfinite"] >= 1
    assert diag["converged"] is False
    assert np.all(np.isfinite(beta))


def test_clean_solve_converges_with_zero_nonfinite():
    """On the same system without poisoning the search converges and records
    no nonfinite evaluation, and the level and margin flags are both set."""
    M, b, Cc, A_inner, K_in, beta_unc, U = _tiny_system()
    beta, diag = C.fit_constrained(M, b, Cc, A_inner, K_in, tau=5.0,
                                   beta_unc=beta_unc, rmax=18.0, r=2)
    assert diag["n_nonfinite"] == 0
    assert diag["converged"] is True
    assert diag["level_met"] and diag["margin_met"]
    for k in ("Lmax", "Fmax", "tau", "tau_requested", "tau_relaxed", "rmax",
              "eps", "rounds", "n_active"):
        assert k in diag
