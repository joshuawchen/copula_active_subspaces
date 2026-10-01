"""Selection of the Stage-1 and Stage-2 multi-index sets (K, q) by held-out
score-matching risk with the one-standard-error rule, and fit_cas_autoselect,
which fits the estimator at the selected sets.
"""
from __future__ import annotations
import numpy as np

from .hermite_score_matching import (rank_gaussianize, enumerate_dictionary,
    build_A_b_streaming, solve_ridge, cas_fit_streaming)

# Candidate grids. The block algebra caps the needed interaction order at 4, so
# no candidate exceeds (K,q) = (4,4); even-degree variants let the selector
# discover the z^3 parity structure. (K, q, even_degree).
STAGE1_GRID = [(4, 2, False), (4, 3, False), (4, 4, False),
               (4, 3, True), (4, 4, True)]
STAGE2_GRID = [(4, 3), (5, 3), (6, 3), (5, 4), (6, 4)]
_PILOT_N = 80_000
_N_SPLITS = 3
_SEED = 20260716


def _samplers():
    from .noise import sample_noise
    from .even_fold import sample_even_fold_noise
    from .conformal_cube import sample_conformal_cube_noise
    return {"banana": sample_noise, "even_fold": sample_even_fold_noise,
            "conformal_cube": sample_conformal_cube_noise}


def _dict(d, K, q, even_degree):
    A = enumerate_dictionary(d, K, q)
    if even_degree:
        A = [a for a in A if sum(a[1]) % 2 == 0]
    return A


def _prune_by_gram(grid, sizes, max_gram_bytes):
    """Drop candidates whose Gram matrix would exceed `max_gram_bytes`.

    The Stage-1 dictionary is a total-degree-K, interaction-order-q truncation
    in d dimensions, so |Lambda| grows like d^q and the (|Lambda|, |Lambda|)
    Gram assembled by build_A_b_streaming grows like d^{2q}; the held-out risk
    holds two of them at once, one for each fold. At d = 40 the (K, q) = (4, 4)
    candidate asks for 137 GB per fold, which is not a slow candidate but an
    impossible one.

    With max_gram_bytes=None nothing is pruned and the selection is exactly the
    published one. The argument exists so a caller working at high d can bound
    the search in advance rather than discover the limit by being killed. The
    leanest candidate is always kept, so the grid is never emptied.
    """
    if max_gram_bytes is None:
        return list(grid)
    keep = [c for c in grid if 8.0 * float(sizes[c]) ** 2 <= max_gram_bytes]
    if not keep:
        keep = [min(grid, key=lambda c: sizes[c])]
    dropped = [c for c in grid if c not in keep]
    if dropped:
        print(f"  [kq_selection] {len(dropped)} candidate(s) over the "
              f"{max_gram_bytes / 2 ** 30:.2f} GB Gram budget, not searched: "
              + ", ".join(f"{c} |Lam|={sizes[c]}" for c in dropped))
    return keep


def _heldout_sm_risk(Z_in, Z_out, A, K):
    """Held-out Hyvarinen risk of the |alpha|^2-ridge score-matching solve.

    Fit theta on Z_in (the deployed |alpha|^2 ridge, solve_ridge), then return
    the UNRIDGED objective 0.5 th^T A_out th - th^T b_out on Z_out. Lower is
    better; the true-score constant is dropped and common to all candidates, so
    risks are comparable across (K,q)."""
    A_in, b_in, _, _, _ = build_A_b_streaming(Z_in, A, K, keep_Phi=False)
    theta, _ = solve_ridge(A_in, b_in, A)
    A_out, b_out, _, _, _ = build_A_b_streaming(Z_out, A, K, keep_Phi=False)
    return float(0.5 * theta @ (A_out @ theta) - theta @ b_out)


def _one_se_select(cands, risks, sizes):
    """1-SE rule: among candidates within one SE of the minimum mean risk, take
    the smallest |Lambda|, then the earliest in `cands`."""
    mean = {c: float(np.mean(risks[c])) for c in cands}
    se = {c: float(np.std(risks[c]) / np.sqrt(len(risks[c]))) for c in cands}
    best = min(cands, key=lambda c: mean[c])
    thr = mean[best] + se[best]
    within = [c for c in cands if mean[c] <= thr]
    sel = min(within, key=lambda c: (sizes[c], cands.index(c)))
    return sel, mean, se


def _print_report(example, stage, grid, report, sel):
    print(f"[{example}] {stage} held-out SM risk (lower better):")
    for c in grid:
        rr = report[c]
        mark = "  <-- SELECT" if c == sel else ""
        print(f"    {str(c):>16}  |Lam|={rr['size']:6d}  "
              f"risk={rr['risk']:+.4f} +/- {rr['se']:.4f}{mark}")


def select_stage1(example, N_pilot=_PILOT_N, n_splits=_N_SPLITS, seed=_SEED,
                  grid=None, verbose=False):
    """Select (K1, q1, even_degree) for `example` by held-out SM risk on the
    full d-dimensional copula score. Returns (selection, report)."""
    grid = grid if grid is not None else STAGE1_GRID
    smp = _samplers()[example]
    risks = {c: [] for c in grid}
    sizes = {}
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        eta = smp(N_pilot, rng)
        n_in = N_pilot // 2
        Z_in = rank_gaussianize(eta[:n_in])
        Z_out = rank_gaussianize(eta[n_in:])
        d = Z_in.shape[1]
        for (K, q, ev) in grid:
            A = _dict(d, K, q, ev)
            sizes[(K, q, ev)] = len(A)
            risks[(K, q, ev)].append(_heldout_sm_risk(Z_in, Z_out, A, K))
    sel, mean, se = _one_se_select(grid, risks, sizes)
    report = {c: {"risk": mean[c], "se": se[c], "size": sizes[c]} for c in grid}
    if verbose:
        _print_report(example, "Stage 1", grid, report, sel)
    return sel, report


def select_stage2(example, stage1, r=4, N_pilot=_PILOT_N, n_splits=_N_SPLITS,
                  seed=_SEED, grid=None, verbose=False):
    """Select (K2, q2) for `example` on the reduced r-space, conditional on the
    Stage-1 selection. The Stage-1 subspace V_r is fit on the in-fold at
    `stage1`, the samples are projected, and the reduced-r held-out SM risk is
    compared across candidates at the fixed |alpha|^2 ridge (the Stage-2
    constants are selected separately downstream). Returns (selection, report)."""
    grid = grid if grid is not None else STAGE2_GRID
    K1, q1, ev1 = stage1
    smp = _samplers()[example]
    risks = {c: [] for c in grid}
    sizes = {}
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        eta = smp(N_pilot, rng)
        n_in = N_pilot // 2
        cas = cas_fit_streaming(eta[:n_in], K=K1, q=q1, r=r, seed=seed,
                                even_degree=ev1)
        V_r = cas["V_r"]
        U_in = cas["Z"] @ V_r
        U_out = rank_gaussianize(eta[n_in:]) @ V_r
        for (K, q) in grid:
            A = enumerate_dictionary(r, K, q)
            sizes[(K, q)] = len(A)
            risks[(K, q)].append(_heldout_sm_risk(U_in, U_out, A, K))
    sel, mean, se = _one_se_select(grid, risks, sizes)
    report = {c: {"risk": mean[c], "se": se[c], "size": sizes[c]} for c in grid}
    if verbose:
        _print_report(example, "Stage 2", grid, report, sel)
    return sel, report


def select_stage_kq(example, N_pilot=_PILOT_N, n_splits=_N_SPLITS, seed=_SEED,
                    r=4, verbose=True):
    """Full per-example selection: Stage-1 (K1,q1,even) then Stage-2 (K2,q2).

    Returns a dict {example, K1, q1, even_degree, K2, q2, stage1_report,
    stage2_report} suitable for populating cas.config.STAGE_KQ."""
    s1, rep1 = select_stage1(example, N_pilot, n_splits, seed, verbose=verbose)
    s2, rep2 = select_stage2(example, s1, r, N_pilot, n_splits, seed,
                             verbose=verbose)
    K1, q1, ev1 = s1
    return {"example": example, "K1": K1, "q1": q1, "even_degree": ev1,
            "K2": s2[0], "q2": s2[1],
            "stage1_report": rep1, "stage2_report": rep2}


# ---------------------------------------------------------------------------
# Per-sample (deployed) selector: uses ONLY the given sample, no pilot/oracle.
# ---------------------------------------------------------------------------
def select_kq_on_sample(eta, r=4, seed=0, n_splits=1,
                        stage1_grid=None, stage2_grid=None,
                        max_gram_bytes=None):
    """Select (K1,q1,even) and (K2,q2) from a SINGLE sample `eta`.

    Both stages are chosen by held-out score-matching risk on an in/out split of
    `eta` itself -- self-contained, using only `eta`, with no pilot and no
    oracle. With n_splits=1 (default) the 1-SE rule reduces to the argmin on one
    random split; larger n_splits averages several splits for stability at the
    cost of more compute. Returns a selection dict {K1,q1,even_degree,K2,q2}."""
    stage1_grid = stage1_grid if stage1_grid is not None else STAGE1_GRID
    stage2_grid = stage2_grid if stage2_grid is not None else STAGE2_GRID
    N = eta.shape[0]
    n_in = N // 2

    # Stage 1: held-out SM risk on the full d-dimensional copula score.
    d = eta.shape[1]
    sizes1 = {c: len(_dict(d, c[0], c[1], c[2])) for c in stage1_grid}
    stage1_grid = _prune_by_gram(stage1_grid, sizes1, max_gram_bytes)
    risks1 = {c: [] for c in stage1_grid}
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        perm = rng.permutation(N)
        Z_in = rank_gaussianize(eta[perm[:n_in]])
        Z_out = rank_gaussianize(eta[perm[n_in:]])
        for (K, q, ev) in stage1_grid:
            A = _dict(d, K, q, ev)
            risks1[(K, q, ev)].append(_heldout_sm_risk(Z_in, Z_out, A, K))
    s1, _, _ = _one_se_select(stage1_grid, risks1, sizes1)
    K1, q1, ev1 = s1

    # Stage 2 conditional on s1: fit V_r on the in-fold at (K1,q1,ev1), project,
    # then select (K2,q2) by held-out reduced-r SM risk at the fixed |alpha|^2
    # ridge (the Stage-2 constants are selected downstream in the density fit).
    sizes2 = {c: len(enumerate_dictionary(r, c[0], c[1])) for c in stage2_grid}
    stage2_grid = _prune_by_gram(stage2_grid, sizes2, max_gram_bytes)
    risks2 = {c: [] for c in stage2_grid}
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(n_splits):
        perm = rng2.permutation(N)
        cas = cas_fit_streaming(eta[perm[:n_in]], K=K1, q=q1, r=r, seed=seed,
                                even_degree=ev1)
        V_r = cas["V_r"]
        U_in = cas["Z"] @ V_r
        U_out = rank_gaussianize(eta[perm[n_in:]]) @ V_r
        for (K, q) in stage2_grid:
            A = enumerate_dictionary(r, K, q)
            risks2[(K, q)].append(_heldout_sm_risk(U_in, U_out, A, K))
    s2, _, _ = _one_se_select(stage2_grid, risks2, sizes2)
    return {"K1": K1, "q1": q1, "even_degree": ev1, "K2": s2[0], "q2": s2[1]}


def fit_cas_autoselect(eta, r=4, seed=0, inner_ridge_scheme=None, M_norm=None,
                       use_secondary_rank=None, n_splits=1, stage1_grid=None,
                       stage2_grid=None, max_gram_bytes=None, **kwargs):
    """Self-contained CAS density estimator (the deployed estimator).

    Selects (K1,q1,even) and (K2,q2) from a held-out split of `eta`
    (select_kq_on_sample), then refits the full ReducedDensityModel at the
    selected dictionaries on ALL of `eta`. Uses only `eta` -- no pilot, no
    oracle. At small N the selection lands on a leaner dictionary (q=2), as the
    sample supports; it sharpens with N. Ridge scheme, M_norm, and
    use_secondary_rank default to cas.config. Returns (model, selection); the
    selection is also attached as model.selected_kq."""
    from .reduced_density import ReducedDensityModel
    from . import config as _cfg
    if inner_ridge_scheme is None:
        inner_ridge_scheme = getattr(_cfg, "RIDGE_DEFAULT",
                                     "theoretical:c_cov=3,c_sob=1e-5")
    if M_norm is None:
        M_norm = getattr(_cfg, "M_NORM", 8192)
    if use_secondary_rank is None:
        use_secondary_rank = getattr(_cfg, "USE_SECONDARY_RANK", False)
    sel = select_kq_on_sample(eta, r=r, seed=seed, n_splits=n_splits,
                              stage1_grid=stage1_grid,
                              stage2_grid=stage2_grid,
                              max_gram_bytes=max_gram_bytes)
    model = ReducedDensityModel(
        r=r, K=sel["K1"], q=sel["q1"], even_degree=sel["even_degree"],
        K_inner=sel["K2"], q_inner=sel["q2"],
        seed=seed, inner_ridge_scheme=inner_ridge_scheme, M_norm=M_norm,
        use_secondary_rank=use_secondary_rank, **kwargs).fit(eta)
    model.selected_kq = sel
    return model, sel
