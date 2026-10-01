"""Constrained Stage-2 solve (Part I, Section 3 and the appendix on the
constrained solve).

Minimizes the Stage-2 quadratic subject to the centering equalities, zero
coefficients above the largest even degree K_eff, and two inequality
families: the leading form F <= -eps on the unit sphere and L <= tau on the
ball of radius rmax. The inequality families are handled by an exchange
method whose numerical searches (max_form_sphere, max_L_ball) check them at
the searched points.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from ._qp_barrier import solve_qp as qp_barrier
from .hermite_score_matching import evaluate_features, evaluate_features_psi_only


def _eval_poly(P, theta, A_inner, K_in):
    from .reduced_density import eval_g_polynomial   # lazy: avoids cycle
    return eval_g_polynomial(P, theta, A_inner, K_in)


# ---------------------------------------------------------------------------
# System assembly (mirrors solve_ridge_theoretical; selected constants passed)
# ---------------------------------------------------------------------------

def assemble_system(Amat, bvec, A_inner, U, N, c_cov, c_sob, K_in):
    """(M, b, Ceq_centering) of the deployed Stage-2 quadratic.

    R = c_cov * sigma_k^2 per degree class + c_sob * ||A||_op * |alpha|^2,
    centering rows C[j, a] = (1/N) sum_n d_j H_a(u_n). The constants are
    the SELECTED pair of the held-out scheme, not hardcoded defaults.
    """
    r = U.shape[1]
    total_deg = np.array([sum(deg) for (_, deg) in A_inner], dtype=int)
    A_op = float(np.linalg.norm(Amat, ord=2))

    R_cov = np.zeros(len(A_inner))
    for k in sorted(set(total_deg.tolist())):
        idx = np.where(total_deg == k)[0]
        R_cov[idx] = c_cov * max(float(np.trace(Amat[np.ix_(idx, idx)])) / N, 0.0)
    R_sob = c_sob * A_op * (total_deg.astype(float) ** 2)
    M = Amat + np.diag(R_cov + R_sob)

    _, Phi_list, _ = evaluate_features(U, A_inner, K_in)
    Cc = np.stack([Phi_list[j].mean(axis=0) for j in range(r)], axis=0)
    del Phi_list
    return M, bvec, Cc


# ---------------------------------------------------------------------------
# Cut machinery (probe-faithful, r-general)
# ---------------------------------------------------------------------------

def sphere_dirs(n, r, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, r))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def initial_cuts(r, seed=0, n_dir=90, radii=(5.0, 8.0, 12.0, 18.0)):
    """Shells outside the data, where an unbounded potential must show itself."""
    return np.vstack([sphere_dirs(n_dir, r, seed) * rad for rad in radii])


def top_degree_rows(A_inner, K_eff):
    """Equality rows zeroing every coefficient of degree > K_eff."""
    idx = [a for a, (_, deg) in enumerate(A_inner) if sum(deg) > K_eff]
    E = np.zeros((len(idx), len(A_inner)))
    for row, a in enumerate(idx):
        E[row, a] = 1.0
    return E, len(idx)


def leading_form_parts(A_inner, K_eff, r):
    """(idx, coef, exps): F(v) = sum coef_j beta_{idx_j} v^{exps_j}."""
    idx, coef, exps = [], [], []
    for a, (supp, deg) in enumerate(A_inner):
        if sum(deg) != K_eff:
            continue
        e = np.zeros(r, dtype=int)
        fac = 1.0
        for k, i in enumerate(supp):
            e[i] = deg[k]
        for d in deg:
            fac *= math.factorial(d)
        idx.append(a)
        coef.append(1.0 / math.sqrt(fac))
        exps.append(e)
    return (np.array(idx, dtype=int), np.array(coef, dtype=float),
            np.array(exps, dtype=int))


def leading_form_rows(A_inner, K_eff, dirs):
    """Rows of the linear map beta -> F(v_k); leading monomial of H_alpha is
    u^alpha / sqrt(alpha!)."""
    rows = np.zeros((len(dirs), len(A_inner)))
    for a, (supp, deg) in enumerate(A_inner):
        if sum(deg) != K_eff:
            continue
        fac = 1.0
        for d in deg:
            fac *= math.factorial(d)
        mono = np.ones(len(dirs))
        for k, i in enumerate(supp):
            mono *= dirs[:, i] ** deg[k]
        rows[:, a] = mono / math.sqrt(fac)
    return rows


def form_vals(dirs, beta, parts):
    idx, coef, exps = parts
    if len(idx) == 0:
        return np.zeros(len(dirs))
    mono = np.prod(dirs[:, None, :] ** exps[None, :, :], axis=2)
    return mono @ (coef * beta[idx])


def max_form_sphere(beta, A_inner, K_eff, parts, r, seed=0, n=200_000, n_ref=5):
    """max_{||v||=1} F(v) and argmax directions: dense search + refinement."""
    dirs = sphere_dirs(n, r, seed)
    F = form_vals(dirs, beta, parts)
    order = np.argsort(-F)[:n_ref]
    best = [dirs[i] for i in order]
    fbest = float(F[order[0]])

    def negF(x):
        v = np.asarray(x, float)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return 1e12
        return -float(form_vals((v / nv)[None, :], beta, parts)[0])

    refined = []
    for v0 in best:
        res = minimize(negF, v0, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-7, "fatol": 1e-10})
        v = res.x / max(np.linalg.norm(res.x), 1e-12)
        refined.append(v)
        fbest = max(fbest, -float(res.fun))
    return fbest, np.array(best + refined)


def max_L_ball(theta, A_inner, K_in, rmax, r, seed=0, n=80_000, n_ref=6):
    """max L over ||u|| <= rmax, the argmax points, and the number of
    nonfinite polynomial values met in the search, grid and refinement
    alike (0 in a sound fit)."""
    rng = np.random.default_rng(seed)
    P = sphere_dirs(n, r, seed) * rng.uniform(0.0, rmax, n)[:, None]
    L = _eval_poly(P, theta, A_inner, K_in)
    finite = np.isfinite(L)
    n_nonfinite = [int((~finite).sum())]
    L = np.where(finite, L, -np.inf)
    order = np.argsort(-L)[:n_ref]
    pts = [P[i] for i in order]
    lbest = float(L[order[0]])

    def f(x):
        if np.linalg.norm(x) > rmax:
            return 1e12
        v = float(_eval_poly(np.asarray(x)[None, :], theta, A_inner, K_in)[0])
        if not np.isfinite(v):
            n_nonfinite[0] += 1
            return 1e12
        return -v

    refined = []
    for p0 in pts:
        res = minimize(f, p0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-9})
        if np.linalg.norm(res.x) <= rmax:
            refined.append(res.x)
            lbest = max(lbest, -float(res.fun))
    return lbest, np.array(pts + refined), n_nonfinite[0]


# ---------------------------------------------------------------------------
# The constrained solve
# ---------------------------------------------------------------------------

def fit_constrained(M, b, Ceq_centering, A_inner, K_in, tau, beta_unc, rmax,
                    r, eps=1e-3, n_rounds=30, relax_level=False, verbose=False):
    """Exchange solve of the Stage-2 quadratic program over the constraint set.

    (M, b, Ceq_centering) is the assembled quadratic (assemble_system), and the
    solve is warm-started from beta_unc. Each round solves the program on the
    current finite cut set and searches for the largest violation of each family
    (max_form_sphere on the unit sphere, max_L_ball on the ball); violating points
    are added as cuts, for the level family together with the points at 0.9 and
    1.1 times its norm. The loop stops when both searches meet their tolerances
    (F <= -eps/2, L <= tau + 1e-3) or after n_rounds, and returns the last
    iterate. relax_level=True raises tau to the attained level when only the level
    family is unmet and repeats the exchange (off by default).

    Returns (beta, diag); diag holds Lmax, Fmax, tau, tau_requested, tau_relaxed,
    rmax, eps, K_eff, n_zeroed, rounds, n_active, n_nonfinite, converged,
    level_met and margin_met. converged requires both tolerances and no nonfinite
    evaluation in the ball search.
    """
    K_eff = K_in if K_in % 2 == 0 else K_in - 1
    E, n_zeroed = top_degree_rows(A_inner, K_eff)
    Ceq = np.vstack([Ceq_centering, E]) if n_zeroed else Ceq_centering
    parts = leading_form_parts(A_inner, K_eff, r)
    tau_requested = float(tau)

    def exchange(tau_level, beta0):
        v_cuts = sphere_dirs(600, r, seed=7)
        u_cuts = initial_cuts(r)
        beta, z, cache = beta0.copy(), None, {}
        Fmax = Lmax = np.inf
        n_active = rnd = n_nonfinite = 0
        converged = ok_F = ok_L = False
        for rnd in range(1, n_rounds + 1):
            G = np.vstack([leading_form_rows(A_inner, K_eff, v_cuts),
                           evaluate_features_psi_only(u_cuts, A_inner, K_in)])
            h = np.concatenate([np.full(len(v_cuts), -eps),
                                np.full(len(u_cuts), tau_level)])
            beta, info = qp_barrier(M, b, Ceq, G, h, z_warm=z, cache=cache)
            z = info["z"]
            n_active = int(info["n_active"])

            Fmax, v_new = max_form_sphere(beta, A_inner, K_eff, parts, r, seed=rnd)
            Lmax, u_new, nnf = max_L_ball(beta, A_inner, K_in, rmax, r, seed=rnd)
            n_nonfinite += nnf
            if verbose:
                print(f"      rnd {rnd:2d}  maxF {Fmax:+.3e}  supL {Lmax:8.2f}  "
                      f"cuts {len(v_cuts)}+{len(u_cuts)}  active {n_active}",
                      flush=True)
            ok_F = Fmax <= -0.5 * eps
            ok_L = Lmax <= tau_level + 1e-3
            if ok_F and ok_L and nnf == 0:
                converged = True
                break
            if ok_F and ok_L:
                # Within tolerance at the finite points, but the search met
                # nonfinite values inside the ball: not a successful search.
                break
            if not ok_F:
                v_cuts = np.vstack([v_cuts, v_new])
            if not ok_L:
                u_cuts = np.vstack([u_cuts, u_new, 0.9 * u_new, 1.1 * u_new])
        return beta, dict(Lmax=float(Lmax), Fmax=float(Fmax), rounds=int(rnd),
                          n_active=int(n_active), n_nonfinite=int(n_nonfinite),
                          converged=bool(converged), ok_F=bool(ok_F),
                          ok_L=bool(ok_L))

    beta, d = exchange(tau, beta_unc)
    tau_used, relaxed = float(tau), False
    if relax_level and not d["converged"] and d["ok_F"] and not d["ok_L"]:
        tau_used = float(d["Lmax"]) + 1e-3
        relaxed = True
        beta, d2 = exchange(tau_used, beta)
        d2["rounds"] += d["rounds"]
        d2["n_nonfinite"] += d["n_nonfinite"]
        d = d2

    diag = {"Lmax": d["Lmax"], "Fmax": d["Fmax"], "tau": tau_used,
            "tau_requested": tau_requested, "tau_relaxed": relaxed,
            "rmax": float(rmax), "eps": float(eps), "K_eff": int(K_eff),
            "n_zeroed": int(n_zeroed), "rounds": d["rounds"],
            "n_active": d["n_active"], "n_nonfinite": d["n_nonfinite"],
            "converged": d["converged"], "level_met": d["ok_L"],
            "margin_met": d["ok_F"]}
    import warnings
    if relaxed:
        warnings.warn(f"constrained solve: level bound relaxed from "
                      f"{tau_requested:.4g} to {tau_used:.4g} to admit the "
                      f"leading-form margin")
    if diag["n_nonfinite"]:
        warnings.warn(f"constrained solve: {diag['n_nonfinite']} nonfinite "
                      f"polynomial values met in the ball search")
    if not diag["converged"]:
        if diag["level_met"] and diag["margin_met"]:
            warnings.warn(f"constrained solve: tolerances met at the finite "
                          f"points but {diag['n_nonfinite']} nonfinite value(s) "
                          f"met in the ball search; convergence not reported")
        else:
            warnings.warn(f"constrained solve reached the round cap ({n_rounds}) "
                          f"with Fmax={diag['Fmax']:+.3e}, "
                          f"Lmax-tau={diag['Lmax'] - tau_used:+.3e}; last iterate "
                          f"returned (margin met: {diag['margin_met']}, level met: "
                          f"{diag['level_met']})")
    return beta, diag


def log_Z(theta, A_inner, K_in, r, seed, m_norm=8192):
    """Scrambled-Sobol QMC log-normalizer of the (bounded) potential.

    No level bound applied: the constrained potential is bounded above by
    construction, so the integrand needs no modification.
    """
    Mp = 1 << int(np.ceil(np.log2(m_norm)))
    U01 = np.clip(qmc.Sobol(d=r, scramble=True, seed=seed).random(Mp),
                  1e-12, 1 - 1e-12)
    p = _eval_poly(norm.ppf(U01), theta, A_inner, K_in)
    mx = float(p.max())
    ev = np.exp(p - mx)
    return mx + float(np.log(ev.mean()))
