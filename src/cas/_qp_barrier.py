#!/usr/bin/env python3
"""Barrier-Newton solver for the constrained Stage-2 quadratic program

    min 1/2 beta' M beta - b' beta   s.t.   Ceq beta = 0,   G beta <= h.

The equalities are eliminated through a null-space basis; the inequalities are
handled by a log barrier with Newton steps from a phase-I start. The active
set is returned with the solution. Run as a script, the module compares the
solver with scipy's trust-constr on random instances.
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space


def _damped_newton(phi, grad, hess, s_of, z, max_it=60, tol=1e-10):
    """Damped Newton with Armijo backtracking, kept inside the barrier domain."""
    for _ in range(max_it):
        g = grad(z)
        H = hess(z)
        try:
            cf = cho_factor(H, lower=True)
        except np.linalg.LinAlgError:
            H = H + (1e-10 * max(np.trace(H), 1.0) / len(H)) * np.eye(len(H))
            cf = cho_factor(H, lower=True)
        dz = -cho_solve(cf, g)
        dec = float(-g @ dz)                       # Newton decrement squared
        if dec / 2.0 <= tol:
            return z
        f0, step = phi(z), 1.0
        while True:
            zn = z + step * dz
            if np.min(s_of(zn)) > 0 and phi(zn) <= f0 - 0.25 * step * dec:
                break
            step *= 0.5
            if step < 1e-14:
                return z
        z = zn
    return z


def _phase1(Gt, h, z0=None, tol=1e-9):
    """Strictly feasible z with Gt z < h, via  min_{z,t} t  s.t.  Gt z - h <= t.

    beta = 0 is NOT feasible: the leading-form rows require F(v) <= -eps < 0 and F
    vanishes at beta = 0.  The augmented problem is strictly feasible at
    (z0, t = max(Gt z0 - h) + 0.1) for ANY z0, so a warm z0 from the previous
    cutting-plane round costs one cheap phase-I rather than a cold restart.
    """
    k, n = Gt.shape[1], Gt.shape[0]
    z0 = np.zeros(k) if z0 is None else np.asarray(z0, float)
    t0 = float(np.max(Gt @ z0 - h)) + 0.1
    x = np.concatenate([z0, [t0]])                 # x = (z, t)

    def s_of(x):                                   # slacks: h + t - Gt z > 0
        return h + x[-1] - Gt @ x[:k]

    tb = 1.0
    for _ in range(60):
        def phi(x, tb=tb):
            s = s_of(x)
            if np.min(s) <= 0:
                return np.inf
            return tb * x[-1] - float(np.sum(np.log(s)))

        def grad(x, tb=tb):
            s = s_of(x)
            inv = 1.0 / s
            g = np.empty(k + 1)
            g[:k] = Gt.T @ inv
            g[-1] = tb - float(inv.sum())
            return g

        def hess(x, tb=tb):
            s = s_of(x)
            w = 1.0 / (s * s)
            H = np.empty((k + 1, k + 1))
            GW = Gt * w[:, None]
            H[:k, :k] = Gt.T @ GW
            H[:k, -1] = -(GW.sum(axis=0))
            H[-1, :k] = H[:k, -1]
            H[-1, -1] = float(w.sum())
            return H + 1e-12 * np.eye(k + 1)

        x = _damped_newton(phi, grad, hess, s_of, x)
        if x[-1] < -1e-9:                          # strictly feasible margin found
            return x[:k]
        if n / tb < tol:
            break
        tb *= 10.0
    if np.min(h - Gt @ x[:k]) > 0:
        return x[:k]
    raise RuntimeError("phase I failed: the constraint set appears empty "
                       "(at odd top degree this is EXPECTED -- no odd-degree "
                       "polynomial is bounded above, so F <= -eps is infeasible)")


def solve_qp(M, b, Ceq, G, h, tol=1e-9, mu=20.0, t0=1.0, z_warm=None,
             cache=None):
    """min 1/2 x'Mx - b'x  s.t.  Ceq x = 0,  G x <= h.  Returns (x, info).

    `cache` is an optional dict reused across cutting-plane rounds.  Ceq and M do
    not change between rounds, so the nullspace basis Nb and the reduced Hessian
    Mt = Nb' M Nb are computed ONCE; only G grows.  Combined with a warm z (and a
    warm-started phase I, since a fresh cut makes the previous z infeasible by
    construction), this is the bulk of the achievable speedup: the per-round cost
    drops to the barrier iterations alone.
    """
    t_start = time.time()
    if cache is not None and "Nb" in cache:
        Nb, Mt, bt = cache["Nb"], cache["Mt"], cache["bt"]
    else:
        Nb = null_space(np.atleast_2d(Ceq))        # (p, k); exact equality handling
        Mt = Nb.T @ M @ Nb
        Mt = 0.5 * (Mt + Mt.T)
        bt = Nb.T @ b
        if cache is not None:
            cache.update({"Nb": Nb, "Mt": Mt, "bt": bt})
    k = Nb.shape[1]
    Gt = G @ Nb
    n = Gt.shape[0]

    s_of = lambda z: h - Gt @ z
    if z_warm is not None and np.min(s_of(z_warm)) > 0:
        z = np.asarray(z_warm, float)
    else:
        z = _phase1(Gt, h, z0=z_warm)              # warm-started phase I

    t = t0
    while True:
        def phi(z, t=t):
            s = s_of(z)
            if np.min(s) <= 0:
                return np.inf
            return t * (0.5 * z @ (Mt @ z) - bt @ z) - float(np.sum(np.log(s)))

        def grad(z, t=t):
            s = s_of(z)
            return t * (Mt @ z - bt) + Gt.T @ (1.0 / s)

        def hess(z, t=t):
            s = s_of(z)
            w = 1.0 / (s * s)
            return t * Mt + Gt.T @ (Gt * w[:, None])

        z = _damped_newton(phi, grad, hess, s_of, z)
        if n / t < tol:
            break
        t *= mu

    x = Nb @ z
    s = s_of(z)
    active = np.where(s < 1e-6 * max(1.0, float(np.max(np.abs(h)))))[0]
    info = {
        "z": z, "k": k, "n_ineq": n,
        "active": active,                # conditional on this, the estimator is
        "n_active": int(active.size),    # LINEAR in b again -- what App E/F needs
        "eq_resid": float(np.abs(np.atleast_2d(Ceq) @ x).max()),
        "max_viol": float(np.max(G @ x - h)),
        "obj": float(0.5 * x @ (M @ x) - b @ x),
        "secs": time.time() - t_start,
    }
    return x, info


def _selftest():
    """Check against scipy trust-constr on random instances of the same shape."""
    from scipy.optimize import minimize, LinearConstraint
    rng = np.random.default_rng(0)
    print(f"{'p':>4} {'m_eq':>5} {'n_ineq':>7} {'barrier(s)':>11} "
          f"{'trust(s)':>9} {'speedup':>8} {'d_obj':>10} {'eq_res':>9} "
          f"{'viol':>9} {'active':>7}")
    for (p, m_eq, n_in) in [(60, 4, 200), (116, 8, 500), (190, 24, 760)]:
        Araw = rng.standard_normal((p + 20, p))
        M = Araw.T @ Araw / (p + 20) + 0.05 * np.eye(p)
        b = rng.standard_normal(p)
        Ceq = rng.standard_normal((m_eq, p))
        G = rng.standard_normal((n_in, p))
        # Build a FEASIBLE instance: pick a reference point inside null(Ceq) and
        # set h so that point is strictly feasible.  (Anchoring h on the
        # unconstrained optimum instead can make the set empty, since nothing
        # forces null(Ceq) to meet it -- phase I correctly reports that.)
        Nb0 = null_space(Ceq)
        x0 = Nb0 @ rng.standard_normal(Nb0.shape[1])
        h = G @ x0 + 0.1                 # x0 strictly feasible; optimum pushes out

        t0 = time.time()
        x1, info = solve_qp(M, b, Ceq, G, h)
        t1 = time.time() - t0

        t0 = time.time()
        res = minimize(lambda x: 0.5 * x @ (M @ x) - b @ x, np.zeros(p),
                       jac=lambda x: M @ x - b, hess=lambda x: M,
                       constraints=[LinearConstraint(Ceq, 0, 0),
                                    LinearConstraint(G, -np.inf, h)],
                       method="trust-constr",
                       options={"maxiter": 400, "gtol": 1e-10, "xtol": 1e-12})
        t2 = time.time() - t0
        o2 = 0.5 * res.x @ (M @ res.x) - b @ res.x
        print(f"{p:>4} {m_eq:>5} {n_in:>7} {t1:>11.3f} {t2:>9.3f} "
              f"{t2 / max(t1, 1e-9):>7.1f}x {info['obj'] - o2:>10.2e} "
              f"{info['eq_resid']:>9.1e} {info['max_viol']:>9.1e} "
              f"{info['n_active']:>7d}")
    print("\n  d_obj = barrier objective - trust-constr objective (<=0 means barrier "
          "found the same or a better point).")
    print("  eq_res, viol: equality residual and worst inequality violation.")


if __name__ == "__main__":
    _selftest()
