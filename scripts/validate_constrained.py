#!/usr/bin/env python3
"""Checks of the constrained Stage-2 solve: assemble_system reproduces
solve_ridge_theoretical with the equalities only; on Gaussian data the
constraints are slack and the solution is unchanged; on Example-1 data both
searches meet their tolerances after the exchange.

Run: PYTHONPATH=src python scripts/validate_constrained.py
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from cas import sample_noise                                    # noqa: E402
from cas.constrained import (assemble_system, fit_constrained,   # noqa: E402
                              log_Z, max_L_ball)
from cas.hermite_score_matching import (build_A_b_streaming,                # noqa: E402
                              enumerate_dictionary,
                              solve_ridge_theoretical)
from cas.reduced_density import (rank_transform_through_marginals,  # noqa: E402
                                  _build_marginals_cached,
                                  eval_g_polynomial)

R, K_IN, Q_IN = 4, 5, 3
C_COV, C_SOB = 3.0, 1e-5
N = 2500
TOL_REL = 1e-6

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}  {detail}")
    if not ok:
        failures.append(name)


def kkt_equality_solve(M, b, Ceq):
    """Closed-form solution of min 0.5 x'Mx - b'x s.t. Ceq x = 0."""
    n = M.shape[0]
    m = Ceq.shape[0]
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = M
    KKT[:n, n:] = Ceq.T
    KKT[n:, :n] = Ceq
    rhs = np.concatenate([b, np.zeros(m)])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:n]


def run_leg(name, U, expect_slack):
    print(f"\n== {name}  (N={U.shape[0]}, r={U.shape[1]}, K_in={K_IN}) ==")
    A_inner = enumerate_dictionary(R, K_IN, Q_IN)
    Amat, bvec, _, _, _ = build_A_b_streaming(U, A_inner, K_IN, keep_Phi=False)
    theta_unc = solve_ridge_theoretical(
        Amat, bvec, A_inner, U_train=U, c_cov=C_COV, c_sob=C_SOB,
        N=U.shape[0])
    if isinstance(theta_unc, tuple):
        theta_unc = theta_unc[0]

    M, b, Cc = assemble_system(Amat, bvec, A_inner, U, U.shape[0],
                               C_COV, C_SOB, K_IN)
    theta_kkt = kkt_equality_solve(M, b, Cc)
    rel0 = (np.linalg.norm(theta_kkt - theta_unc)
            / max(np.linalg.norm(theta_unc), 1e-12))
    check("(0) assembly fidelity", rel0 < 1e-6, f"rel diff {rel0:.2e}")

    tg = eval_g_polynomial(U, theta_unc, A_inner, K_IN)
    tau = float(np.percentile(tg, 99.9))
    rmax = max(18.0, 1.5 * float(np.linalg.norm(U, axis=1).max()))
    supL_unc, _ = max_L_ball(theta_unc, A_inner, K_IN, rmax, R, seed=99)

    t0 = time.time()
    beta, diag = fit_constrained(M, b, Cc, A_inner, K_IN, tau, theta_unc,
                                 rmax, R)
    dt = time.time() - t0
    print(f"      tau {tau:+.3f}  supL_unc {supL_unc:+.2f}  "
          f"rounds {diag['rounds']}  active {diag['n_active']}  "
          f"zeroed {diag['n_zeroed']}  K_eff {diag['K_eff']}  {dt:.1f}s")

    check("(ii) supL <= tau + 1e-3", diag["Lmax"] <= tau + 1e-3,
          f"Lmax {diag['Lmax']:+.4f}")
    check("(ii) Fmax <= -eps/2", diag["Fmax"] <= -0.5e-3,
          f"Fmax {diag['Fmax']:+.3e}")

    rel = (np.linalg.norm(beta - theta_unc)
           / max(np.linalg.norm(theta_unc), 1e-12))
    if expect_slack and supL_unc <= tau + 1e-3:
        check("(i) relaxation equality (slack case)", rel < 1e-3,
              f"rel diff {rel:.2e}")
    else:
        print(f"      (i) bound binds here (supL_unc {supL_unc:+.2f} > "
              f"tau {tau:+.2f}): rel move {rel:.2e} -- expected, not a check")

    lz = log_Z(beta, A_inner, K_IN, R, seed=0)
    check("(iv) log_Z finite (no level bound)", np.isfinite(lz),
          f"log_Z {lz:+.4f}")
    return beta, Cc, A_inner, diag, rmax


def run_relaxation_leg(beta_star, Cc, A_inner, diag_star, rmax):
    """Check (i) on a problem where the unconstrained optimum IS feasible.

    Legs A and B never exercise this: at production scale the sup bound
    binds in both (supL_unc ~ 1e4 against tau ~ 1-2), so the branch that
    would verify 'the constrained solve returns the unconstrained optimum
    when the constraint is slack' is skipped and only reports a move. That
    leaves the property untested, and it is the one a sign error or a
    mis-assembled constraint row would break while feasibility and assembly
    fidelity both still passed.

    Construction. Take beta_star = the constrained solution of Leg A. It
    already satisfies the equality rows (it came out of that QP) and is
    strictly feasible at a margin below the fit default. Set M = I and
    b = beta_star, so the unconstrained minimizer subject to those
    equalities is exactly beta_star; give the sup bound slack (tau = its own
    Lmax + 1) and a margin eps small enough that its leading form clears it.
    The solver must then return beta_star unmoved.
    """
    print("\n== Leg C: designed-feasible optimum (relaxation regime) ==")
    n = beta_star.shape[0]
    M = np.eye(n)
    b = beta_star.copy()

    tau_slack = float(diag_star["Lmax"]) + 1.0
    # Leg A terminates at Fmax <= -eps/2, so its leading form clears a
    # margin of eps/2 but not necessarily eps. Test at a margin it clears.
    eps_test = 1e-4
    check("(i) beta_star clears the test margin",
          diag_star["Fmax"] <= -eps_test,
          f"Fmax {diag_star['Fmax']:+.3e} vs -eps_test {-eps_test:+.1e}")

    beta, diag = fit_constrained(M, b, Cc, A_inner, K_IN, tau_slack,
                                 beta_star, rmax, R, eps=eps_test)
    rel = (np.linalg.norm(beta - beta_star)
           / max(np.linalg.norm(beta_star), 1e-12))
    print(f"      tau_slack {tau_slack:+.3f}  supL {diag['Lmax']:+.3f}  "
          f"rounds {diag['rounds']}  active {diag['n_active']}")
    check("(i) relaxation equality: feasible optimum is returned unmoved",
          rel < 1e-6, f"rel diff {rel:.2e}")


def main():
    print("validate-constrained: sup-constrained Stage-2 gate checks")

    rng = np.random.default_rng(0)
    U_gauss = rng.standard_normal((N, R))
    beta_A, Cc_A, A_A, diag_A, rmax_A = run_leg(
        "Leg A: standard-Gaussian U (slack regime)", U_gauss,
        expect_slack=True)

    eta = sample_noise(N, np.random.default_rng(1000))
    marg = _build_marginals_cached(eta)
    Z = rank_transform_through_marginals(eta, marg)
    rngV = np.random.default_rng(7)
    Vr, _ = np.linalg.qr(rngV.standard_normal((Z.shape[1], R)))
    run_leg("Leg B: Example-1 noise, rank-transformed projection", Z @ Vr,
            expect_slack=False)

    run_relaxation_leg(beta_A, Cc_A, A_A, diag_A, rmax_A)

    print()
    if failures:
        print(f"FAIL ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
