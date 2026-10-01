#!/usr/bin/env python3
"""
check_paper_numbers.py -- keep the paper's cache-derived numbers in sync
with the committed caches, with a before/after audit and narrative guards.

WHY THIS EXISTS
---------------
Every experimental number in the paper is a function of a committed cache
in ``cache/``.  Hand-typed numbers drift from the caches after a rerun.
This tool makes the caches the single source of truth: it recomputes each
claim *exactly* the way the corresponding figure script does, compares it
to what is currently written in the .tex, and (with ``--apply``) rewrites
only the numeric tokens -- never the prose.

THE NARRATIVE PROBLEM
---------------------
A number change sometimes invalidates the surrounding *story* (e.g.
"enriching to K1=5 reduces the floor" stops being true if the floor no
longer drops; "2-3x improvement" breaks if a factor leaves that range).
Pure find-and-replace would silently keep stale prose around a fresh
number.  So every claim carries a ``narrative_guard``: a check on the
(before, after) values that flags when the prose logic may no longer
hold.  Flagged claims are NOT auto-applied; they are printed for review
and only changed with ``--apply-narrative KEY``.

USAGE
-----
    python3 scripts/check_paper_numbers.py                 # audit (read-only)
    python3 scripts/check_paper_numbers.py --apply         # apply mechanical updates
    python3 scripts/check_paper_numbers.py --apply-narrative sec51_floor
    python3 scripts/check_paper_numbers.py --figures       # also regenerate figures
    python3 scripts/check_paper_numbers.py --only sec52 sec53

Exit code is non-zero if any STALE or NARRATIVE-flagged claim remains
(useful as a CI / pre-commit check).

ADDING A CLAIM
--------------
Append a ``Claim`` to ``REGISTRY``.  Give it (a) a regex that matches the
sentence/table-row exactly once with one named group per number, (b) a
``compute`` that returns {group: float} from the caches, (c) a ``fmt`` per
group (a format string like "{:.2f}" or a callable value->str for custom
rendering, e.g. a floor/ceil clamp for a stated-range endpoint), and
(d) optionally a ``narrative_guard``.

Coverage: every cache-derived number in the four documents: Part I main
(spectrum, sec:exp-banana, sec:exp-testll, sec:exp-bip); Part I supplement
(SM1 PCA tables, SM3 regularizer and c-stability tables, the cross-example
regularizer table, the ehat-tightness table, the centering tables and
paragraph); Part II main (sec:tightness, the a priori and a posteriori
sentences and captions); Part II supplement (tab:sm-jhi and its
Interpretation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, "cache")
MAIN = os.path.join(REPO, "paper", "part1_method.tex")
SUPP = os.path.join(REPO, "paper", "part1_supplement.tex")
P2MAIN = os.path.join(REPO, "paper", "part2_analysis.tex")
P2SUPP = os.path.join(REPO, "paper", "part2_supplement.tex")

N_HEADLINE = 50000          # headline sample size (matches main text)
R_ORACLE = 4                # deployed rank r
B_SUPPORT = 8               # coordinates the orthogonal mixing touches == copula rank


# --------------------------------------------------------------------------
# Cache-derived computations.  Each MIRRORS the named figure script so the
# prose can never disagree with the plotted value.  Keep these in lockstep
# with figures/ if a figure's formula ever changes.
# --------------------------------------------------------------------------
def _jload(name: str) -> dict:
    with open(os.path.join(CACHE, name)) as f:
        return json.load(f)


def _spectrum_from_analytic(tag: str) -> dict:
    """Exact score-covariance spectrum for one example.  Source: the
    analytic-oracle eigendecomposition analytic_VrC_{tag}_* (cas.
    analytic_oracle): the exact C evaluated by Monte Carlo at N_ref=1e6 from
    the closed-form copula score, which is the reference sec:exp-benchmark
    states.  Only the top-8 eigenpairs are stored, so the cliff diagnostic
    runs only when eigenvalues beyond B_SUPPORT are present."""
    z = np.load(os.path.join(
        CACHE, f"analytic_VrC_{tag}_top8_Nref1000000_seed20260715.npz"))
    ev = np.asarray(z["eigvals_top"], dtype=float)
    trC = float(z["trC"])
    top4 = ev[:R_ORACLE]
    tail = trC - float(top4.sum())          # bottom-(d-r) sum
    out = {"e1": top4[0], "e2": top4[1], "e3": top4[2], "e4": top4[3],
           "tail": tail}
    # Cliff diagnostics, for the guard only (not written into the .tex).
    if len(ev) > B_SUPPORT:
        out["_cliff"] = float(ev[B_SUPPORT - 1] / max(ev[B_SUPPORT], 1e-12))
        out["_beyond_b"] = float(trC - float(ev[:B_SUPPORT].sum()))
    return out


def compute_spectrum() -> dict:
    return _spectrum_from_analytic("banana")


def compute_spectrum_ex2() -> dict:
    return _spectrum_from_analytic("even_fold")


def compute_spectrum_ex3() -> dict:
    return _spectrum_from_analytic("conformal_cube")


def _frob_sin_to_deg(s: float, r: int = R_ORACLE) -> float:
    """Per-direction principal angle in degrees, as plotted in fig03."""
    return math.degrees(math.asin(min(s / math.sqrt(r), 1.0)))


def _sec51_run2026() -> dict:
    """Per-example, per-N seed records of the run2026 subspace caches:
    {tag: {N: {"cas"/"pca"/"rand": {seed: sinTheta_F},
               "kq": {seed: (q1, even_degree)}}}}."""
    out = {}
    for tag in ("banana", "even_fold", "conformal_cube"):
        out[tag] = {}
        for N in (500, 1000, 2000, 5000, 10000, 20000, 50000):
            d = _run2026(tag, "subspace", N)
            out[tag][N] = {
                "cas": {s: float(v) for s, v in d["CAS"].items()},
                "pca": {s: float(v) for s, v in d["PCA"].items()},
                "rand": {s: float(v) for s, v in d["random"].items()},
                "kq": {s: (int(v["q1"]), bool(v["even_degree"]))
                       for s, v in d["selected_kq"].items()},
            }
    return out


def compute_floor() -> dict:
    """§5.1 subspace-recovery prose: the seed count and the selected
    interaction orders stated in the paragraph, from the run2026 subspace
    caches; the paragraph's non-numeric claims (N-independent PCA/random,
    order-of-magnitude drop at the selection crossing, unanimity at the
    largest N, mixed per-seed selection in the crossover region) are
    defended by guard_floor."""
    data = _sec51_run2026()
    seeds = min(len(cell["cas"]) for per_n in data.values()
                for cell in per_n.values())
    qlow = {q for tag in data for (q, _e) in data[tag][500]["kq"].values()}
    if len(qlow) != 1:
        raise ValueError(f"selection at N=500 not unanimous across examples: {qlow}")
    q50 = {tag: set(data[tag][N_HEADLINE]["kq"].values()) for tag in data}
    for tag, s in q50.items():
        if len(s) != 1:
            raise ValueError(f"{tag}: selection at N={N_HEADLINE} not unanimous: {s}")
    (q_b, _), = q50["banana"]
    (q_e, _), = q50["even_fold"]
    (q_c, ev_c), = q50["conformal_cube"]
    if q_b != q_e:
        raise ValueError(f"Examples 1-2 disagree at N={N_HEADLINE}: {q_b} vs {q_e}")
    if not ev_c:
        raise ValueError("Example 3 headline selection is not the even-degree dictionary")
    return {"seeds": float(seeds), "qlow": float(next(iter(qlow))),
            "qhi12": float(q_b), "qhi3": float(q_c)}


def _run2026(tag: str, kind: str, N: int) -> dict:
    return _jload(os.path.join("run2026", tag, f"{kind}_N{N}.json"))


def _section52_kls(bench: str = "banana") -> dict:
    """Per-method noise KL at the headline N: seed means of the run2026
    testll cache, which stores the held-out noise KL per seed directly.
    Mirrors fig04 on the regenerated pipeline."""
    d = _run2026(bench, "testll", N_HEADLINE)
    m = lambda k: float(np.nanmean([v for v in d[k].values() if v is not None]))
    return {"cas": m("CAS"), "pca": m("PCA"), "pom": m("PoM"),
            "gauss": m("Gaussian")}


def compute_section52() -> dict:
    return _section52_kls()


def _section53_kls() -> dict:
    """Posterior KL at the headline N: seed means of the run2026 posterior
    cache.  Mirrors fig06 on the regenerated pipeline."""
    c = _run2026("banana", "posterior", N_HEADLINE)
    m = lambda k: float(np.nanmean([v for v in c[k].values() if v is not None]))
    return {"cas": m("CAS"), "pca": m("PCA"), "pom": m("PoM"),
            "gauss": m("Gaussian")}


def compute_section53() -> dict:
    return _section53_kls()


# ----- Supplement: sharp-bound J_hi table (tab:sm-jhi) + its Interpretation --
# Three-source assembly:
#   trace bound = rank_price from cache/stage1_cert_{law}.json (the exact-C
#                 spectrum evaluation, the same value sec:tightness prints);
#   gap         = (1/2) sum_i psi(lambda_i(F_perp)), psi(l) = l - 1 - log l,
#                 from the H_perp eigenvalues of cache/jhi_supplement.json
#                 (the cancellation-free form the table caption states);
#   J_hi        = trace - gap (eq:sm-jhi-trace-gap, an identity);
#   DKL         = delta_sub_Z_mean/se from cache/delta_sub_Z_{law}.json
#                 (the sm:delta-sub-Z quadrature at V_r^C);
#   sinTheta    = sin_theta from cache/jhi_supplement.json.
# The jhi cache's direct J_hi_at_* and true_kl_at_* fields are NOT read: the
# direct difference of the two bounds loses three significant figures to the
# shared tr(Sigma_perp - I) cancellation, and the true_kl stage was computed
# under a superseded transform.
_JHI_LAW = {"example1": "banana", "example2": "even_fold",
            "example3": "conformal_cube"}


def _jhi_cache() -> dict:
    return json.load(open(os.path.join(CACHE, "jhi_supplement.json")))


def _rank_price(law: str) -> float:
    return float(_jload(f"stage1_cert_{law}.json")["rank_price"])


def _psi_gap(label: str) -> float:
    ev = np.asarray(_jhi_cache()[label]["H_perp_eigs"], dtype=float)
    return 0.5 * float(np.sum(ev - 1.0 - np.log(ev)))


def _sci1(x: float) -> tuple:
    """Mantissa/exponent for the table's a.b\\cdot10^{-e} gap cells, the
    mantissa rounded to 1 decimal and renormalized to stay in [1, 10)."""
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    if round(m, 1) >= 10.0:
        m, e = m / 10.0, e + 1
    return m, float(-e)


def _jhi_pct() -> float:
    """The smallest one-decimal percentage strictly above the largest
    gap/trace ratio: the self-maintaining ceiling for every 'less than X%'
    site of the refinement family (supplement Interpretation + the three
    Part II main-text sentences)."""
    worst = max(_psi_gap(f"example{ex}") / _rank_price(_JHI_LAW[f"example{ex}"])
                for ex in "123")
    return math.ceil(1000.0 * worst) / 10.0


def _jhi_eigrange() -> tuple:
    evs = np.concatenate([np.asarray(_jhi_cache()[f"example{ex}"]["H_perp_eigs"],
                                     dtype=float) for ex in "123"])
    lo = math.floor(1000.0 * float(evs.min())) / 1000.0
    hi = math.ceil(1000.0 * float(evs.max())) / 1000.0
    dep = math.ceil(1000.0 * float(np.max(np.abs(evs - 1.0)))) / 1000.0
    return lo, hi, dep


def _jhi_row(label: str) -> dict:
    law = _JHI_LAW[label]
    rp, gap = _rank_price(law), _psi_gap(label)
    ds = _jload(f"delta_sub_Z_{law}.json")
    gm, ge = _sci1(gap)
    return {"trace": rp, "jhi": rp - gap, "gm": gm, "ge": ge,
            "dm": float(ds["delta_sub_Z_mean"]), "ds": float(ds["delta_sub_Z_se"]),
            "st": float(_jhi_cache()[label]["sin_theta"])}


def compute_jhi_row1() -> dict:
    return _jhi_row("example1")


def compute_jhi_row2() -> dict:
    return _jhi_row("example2")


def compute_jhi_row3() -> dict:
    return _jhi_row("example3")


def compute_jhi_reading() -> dict:
    rows = {ex: _jhi_row(f"example{ex}") for ex in "123"}
    lo, hi, dep = _jhi_eigrange()
    st = math.ceil(1000.0 * max(rows[ex]["st"] for ex in "123")) / 1000.0
    out = {"pct": _jhi_pct(), "lo": lo, "hi": hi, "dep": dep, "st": st}
    for ex in "123":
        out[f"f{ex}"] = rows[ex]["trace"] / rows[ex]["dm"]
        out[f"j{ex}"] = rows[ex]["jhi"] / rows[ex]["dm"]
    return out


# ----- Supplement SM1: PCA-Vr ablation tables (tab:pca-banana, tab:pca-bip) --
# Source: cache/sm1_pca_banana.json + cache/sm1_pca_bip.json, produced by
# experiments/exp_sm1_pca_ablation.py.  The table-row claims are built in a
# loop after the REGISTRY definition (the rows are uniform).
_SM1_NS = (500, 1000, 2000)


def _sm1_banana_row(r: int) -> dict:
    d = _jload("sm1_pca_banana.json")[str(r)]
    out = {}
    for N in _SM1_NS:
        out[f"cas{N}"] = float(np.mean(d[str(N)]["cas"]))
        out[f"pca{N}"] = float(np.mean(d[str(N)]["pca"]))
    return out


def _sm1_bip_row(name: str) -> dict:
    d = _jload("sm1_pca_bip.json")
    g = float(np.nanmean(d["Gaussian"]))
    a = np.array(d[name], dtype=float)
    m = float(np.nanmean(a))
    out = {"mean": m, "median": float(np.nanmedian(a)), "worst": float(np.nanmax(a))}
    if name != "Gaussian":
        out["red"] = 100.0 * (g - m) / g
    return out


def _pct(x: float) -> str:
    """Render a reduction percentage as the paper does: '-6', '62' (no '+')."""
    return f"{round(x):d}"


def guard_sm1_banana(before: dict, after: dict, r: int = 0) -> list[str]:
    # Caption / prose: "CAS is better in 12 of the 15 pairs; the other three,
    # at N=500 for r in {2, 3, 8}, are within the margin". Every row defends
    # its two N >= 1000 wins; rows 4 and 6 also claim the N=500 cell, rows 2,
    # 3 and 8 claim it is within the margin.
    msgs = []
    for N in (1000, 2000):
        if after[f"cas{N}"] - after[f"pca{N}"] < 0.05:
            msgs.append(f"CAS no longer beats PCA by >=0.05 nat at N={N} on the r={r} row; "
                        f"caption claims CAS wins every N>=1000 cell")
    gap500 = after["cas500"] - after["pca500"]
    if r in (4, 6) and gap500 < 0.05:
        msgs.append(f"CAS no longer beats PCA by >=0.05 nat at N=500 on the r={r} row; "
                    f"caption counts that cell among the 12 wins")
    if r in (2, 3, 8) and gap500 >= 0.05:
        msgs.append(f"CAS now beats PCA by >=0.05 nat at N=500 on the r={r} row; "
                    f"caption says that cell is within the margin")
    return msgs


def guard_sm1_bip_cas(before: dict, after: dict) -> list[str]:
    # Prose: CAS "reduces the KL divergence by 50%" (and PCA by at most 3%).
    if after.get("red", 0.0) < 40.0:
        return [f"CAS BIP reduction now {after.get('red', float('nan')):.0f}% (<40%); "
                f"prose says 'reduces the KL divergence by 50%'"]
    return []


# ----- cor44 per-N assembly (feeds _ref_dsub and the sec:tightness claims) --
# Per (benchmark, N), from committed caches:
#   meas   = mean over seeds of the delta-sub-cells quadrature divergence
#   cor44  = mean over seeds of 0.5*(sqrt(E_r)+T2+sqrt(2 trC)*sinTheta)^2
#            (E_r = trC - sum(eigvals_top) from ref; T2 = sqrt(basis_truncation
#            _sq_estimate) from saturation; sinTheta from scan)
#   tight  = mean over seeds of scan oracle_KL_bound = 0.5 tr((I-VV^T)C_Lambda)
_COR44_R = {"banana": 4, "even_fold": 4, "conformal_cube": 4}
_cor44_byN_cache: dict = {}


def _cor44_byN(b: str) -> dict:
    if b not in _cor44_byN_cache:
        import collections
        r = _COR44_R[b]
        ref = _jload(f"cor44_{b}_ref_K4_q2_N1000000.json")
        sat = _jload(f"cor44_{b}_saturation_production_N1000000.json")
        scan = _jload(f"cor44_{b}_scan_K4_q2_r{r}.json")
        dis = _jload(f"delta_sub_Z_cells_{b}.json")["cells"]
        trC = float(ref["trC"]); E_r = trC - float(np.sum(ref["eigvals_top"]))
        T2 = math.sqrt(max(float(sat["basis_truncation_sq_estimate"]), 0.0))
        dlk = {(d["N"], d["seed"]): d["delta_sub_Z_mean"] for d in dis}
        byN = collections.defaultdict(list)
        for c in scan:
            byN[c["N"]].append(c)
        out = {}
        for N, cells in byN.items():
            sts = np.array([c["sin_theta_F"] for c in cells])
            orac = np.array([c["oracle_KL_bound"] for c in cells])
            rhs = 0.5 * (math.sqrt(max(E_r, 0.0)) + T2
                         + math.sqrt(2 * trC) * sts) ** 2
            meas = [dlk[(N, c["seed"])] for c in cells if (N, c["seed"]) in dlk]
            out[N] = {"meas": float(np.mean(meas)) if meas else float("nan"),
                      "cor44": float(rhs.mean()), "tight": float(orac.mean())}
        _cor44_byN_cache[b] = out
    return _cor44_byN_cache[b]


# ----- Part II sec:tightness prose claims ----------------------------------
# Sources: delta_sub_Z_{law}.json (quadrature divergence at V_r^C),
# stage1_cert_{law}.json (rank_price = the exact-spectrum trace bound; per-cell
# C1_exact at the estimated bases), delta_sub_Z_cells_banana.json (per-cell
# quadrature divergence at the estimated bases).
_TIGHT_LAWS = ("banana", "even_fold", "conformal_cube")


def compute_cor44_looseness() -> dict:
    out = {}
    for i, law in enumerate(_TIGHT_LAWS, start=1):
        d = _jload(f"delta_sub_Z_{law}.json")
        rp = _rank_price(law)
        out[f"d{i}"] = float(d["delta_sub_Z_mean"])
        out[f"s{i}"] = float(d["delta_sub_Z_se"])
        out[f"b{i}"] = rp
        out[f"f{i}"] = rp / float(d["delta_sub_Z_mean"])
    return out


def guard_cor44_looseness(before: dict, after: dict) -> list[str]:
    """Bound above measured at the exact basis."""
    msgs = []
    for i in (1, 2, 3):
        if after[f"d{i}"] + 4 * after[f"s{i}"] > after[f"b{i}"]:
            msgs.append(f"Example~{i}: measured divergence {after[f'd{i}']:.4f} "
                        f"(+4 SE) reaches the trace bound {after[f'b{i}']:.4f}; "
                        f"check the quadrature cells for outlier u's before "
                        f"applying")
    return msgs


def _tight_cells_ratio() -> tuple:
    """Per-N ratios of seed means (the printed range) and per-cell ratios
    (the guard's every-realization check), over shared (N>=2500, seed) cells."""
    cert = {(c["N"], c["seed"]): float(c["C1_exact"])
            for c in _jload("stage1_cert_banana.json")["cells"]}
    div = {(c["N"], c["seed"]): float(c["delta_sub_Z_mean"])
           for c in _jload("delta_sub_Z_cells_banana.json")["cells"]}
    keys = sorted(k for k in cert if k in div and k[0] >= 2500)
    if not keys:
        raise SystemExit("no shared (N>=2500, seed) cells between "
                         "stage1_cert_banana and delta_sub_Z_cells_banana")
    per_cell = [cert[k] / div[k] for k in keys]
    ns = sorted({k[0] for k in keys})
    per_n = [float(np.mean([cert[k] for k in keys if k[0] == N]))
             / float(np.mean([div[k] for k in keys if k[0] == N])) for N in ns]
    return per_n, per_cell


def compute_cor44_ratio_range() -> dict:
    per_n, _per_cell = _tight_cells_ratio()
    return {"rlo": math.floor(100.0 * min(per_n)) / 100.0,
            "rhi": math.ceil(100.0 * max(per_n)) / 100.0}


def guard_cor44_ratio_range(before: dict, after: dict) -> list[str]:
    _per_n, per_cell = _tight_cells_ratio()
    if min(per_cell) < 1.0:
        return [f"a per-cell certificate/divergence ratio of {min(per_cell):.2f} "
                f"(<1) at some N>=2500 cell: the certificate no longer bounds "
                f"the measured divergence there; check that cell's quadrature "
                f"for outlier u's before applying"]
    return []


def compute_p2_tight_jhipct() -> dict:
    return {"pct": _jhi_pct()}


def compute_p2_tight_eigrange() -> dict:
    lo, hi, _dep = _jhi_eigrange()
    return {"lo": lo, "hi": hi, "pct": _jhi_pct()}


def compute_p2_lsi_repeats() -> dict:
    """The log-Sobolev looseness range (min and max of the three trace-bound
    to divergence ratios) and the sinTheta tolerance, for the sites that
    repeat the values cor44_slack_looseness and jhi_reading print, so a
    regeneration moves every copy together."""
    rows = {ex: _jhi_row(f"example{ex}") for ex in "123"}
    ratios = [rows[ex]["trace"] / rows[ex]["dm"] for ex in "123"]
    st = math.ceil(1000.0 * max(rows[ex]["st"] for ex in "123")) / 1000.0
    return {"lo": min(ratios), "hi": max(ratios), "st": st}


# ----- Part II: Stage-1 a priori terms + a posteriori estimate --------------
# Source: cache/aposteriori_{tag}.json (experiments/exp_stage1_aposteriori.py;
# hermite_sq copied there from cor44_{tag}_saturation_production_N1000000.json).
# Mirrors figures/fig_stage1_bounds.py: panel-(a) slices are sqrt(hermite_sq),
# sqrt(E_r), sqrt(mean penalty); panel-(b) curves are per-N means of rhs /
# resB / hatEB over the 8 seeds.
_APOST_TAGS = ("banana", "even_fold", "conformal_cube")


def _ref_dsub(tag: str) -> float:
    """Reference Delta_sub at the largest N, in nats: the value reported in
    sec:exp-composite and drawn as the rule in fig:stage1-bounds(a).  Read
    from delta_sub_Z_cells_{tag}.json, the deterministic inner-quadrature
    estimator of Supplement sm:disintegration-mc, and NOT from the cor44
    conditional Monte Carlo cache: that holds the superseded conditional Monte Carlo,
    whose importance weights degenerate on conformal_cube and whose mean is
    destroyed by outliers."""
    byN = _cor44_byN(tag)
    return byN[max(byN)]["meas"]


_apost_cache: dict = {}


def _apost(tag: str) -> dict:
    if tag not in _apost_cache:
        _apost_cache[tag] = _jload(f"aposteriori_{tag}.json")
    return _apost_cache[tag]


def _apost_row(tag: str, N: int) -> dict:
    return next(r for r in _apost(tag)["rows"] if r["N"] == N)


def _apriori_bound_parts() -> list:
    """(bound, reference Delta_sub, Hermite-term nats) per example at N=50000,
    mirroring fig_stage1_bounds panel (a) squared and halved: bound =
    0.5*(sqrt(hermite_sq) + sqrt(E_r) + sqrt(mean penalty))^2."""
    out = []
    for t in _APOST_TAGS:
        d = _apost(t)
        herm = math.sqrt(float(d["hermite_sq"]))
        rank = math.sqrt(float(d["E_r"]))
        pen = max(float(np.mean(_apost_row(t, 50000)["penalty"])), 0.0)
        bound = 0.5 * (herm + rank + math.sqrt(pen)) ** 2
        out.append((bound, _ref_dsub(t), 0.5 * herm ** 2))
    return out


def compute_p2_apriori_loose() -> dict:
    r"""Part II \S7.3: the floor/ceil bracket of the three bound-to-reference
    ratios at N=50000. guard_p2_apriori_loose defends the 'more than half of
    it from the Hermite term alone' relation on every regeneration."""
    ratios = [b / d for b, d, _h in _apriori_bound_parts()]
    return {"rlo": math.floor(10.0 * min(ratios)) / 10.0,
            "rhi": math.ceil(10.0 * max(ratios)) / 10.0}


def guard_p2_apriori_loose(before: dict, after: dict) -> list[str]:
    msgs = []
    for i, (b, _d, h) in enumerate(_apriori_bound_parts(), start=1):
        if h <= 0.5 * b:
            msgs.append(f"Example~{i}: Hermite contribution {h:.2f} nats is no "
                        f"longer more than half of the bound {b:.2f}; the "
                        f"sentence says 'the Hermite term alone being more than half of the bound'")
    return msgs


def _e2e_indices() -> dict:
    """{(example i, N slot j): effectivity index}, i over _APOST_TAGS and j over
    (500, 2500, 12500, 50000), mirroring fig_apost_endtoend: estimate =
    dhat_marg + 0.5*mean(rhs) + dhat_den over reference ref_full_kl."""
    out = {}
    for i, t in enumerate(_APOST_TAGS, start=1):
        e2e = _jload(f"apost_e2e_{t}.json")
        for j, N in enumerate((500, 2500, 12500, 50000), start=1):
            cs = [r for r in e2e if r["N"] == N]
            row = _apost_row(t, N)
            est = (float(np.mean([r["dhat_marg"] + r["log_mass"] for r in cs]))
                   + 0.5 * float(np.mean(row["rhs"]))
                   + float(np.mean([r["dhat_den"] for r in cs])))
            ref = float(np.mean([r["ref_full_kl"] for r in cs]))
            out[(i, j)] = est / ref
    return out


def compute_p2_e2e_eff() -> dict:
    r"""Part II \S7.4: the exception index (Example 3 at N=50000), the only
    index printed; guard_p2_e2e_eff defends the paragraph's relations
    (monotone decrease with N, the above-100 and below-4 thresholds, 11 of 12
    with this one exception) on every regeneration."""
    return {"e34": _e2e_indices()[(3, 4)]}


def guard_p2_e2e_eff(before: dict, after: dict) -> list[str]:
    idx = _e2e_indices()
    msgs = []
    below = sorted(k for k, v in idx.items() if v < 1.0)
    if below != [(3, 4)]:
        msgs.append(f"cells with index < 1 are now {below}; the paragraph says "
                    f"11 of the 12 cells with the one exception at Example 3, "
                    f"N=50000")
    if min(idx[(i, 1)] for i in (1, 2, 3)) <= 100.0:
        msgs.append("an index at N=500 is now <= 100; the paragraph says "
                    "'from the hundreds at N = 500'")
    if max(idx[(i, 1)] for i in (1, 2, 3)) >= 1000.0:
        msgs.append("an index at N=500 is now >= 1000; the paragraph says "
                    "'from the hundreds at N = 500'")
    if max(idx[(i, j)] for i in (1, 2, 3) for j in (3, 4)) >= 4.0:
        msgs.append("an index at N >= 12500 is now >= 4; the paragraph says "
                    "'below 4 from N >= 12,500'")
    for i, t in enumerate(_APOST_TAGS, start=1):
        # sec:limitations: at N = 500 the bounded term alone exceeds the
        # reference total by a factor greater than 100 on every example.
        e2e = [r for r in _jload(f"apost_e2e_{t}.json") if r["N"] == 500]
        bounded = 0.5 * float(np.mean(_apost_row(t, 500)["rhs"]))
        ref = float(np.mean([r["ref_full_kl"] for r in e2e]))
        if bounded / ref <= 100.0:
            msgs.append(f"Example {i}: bounded term / reference total at N=500 "
                        f"is {bounded / ref:.0f}; sec:limitations says 'a factor "
                        f"greater than 100 on every example'")
    for i in (1, 2, 3):
        vs = [idx[(i, j)] for j in (1, 2, 3, 4)]
        if any(b >= a for a, b in zip(vs, vs[1:])):
            msgs.append(f"Example {i} indices {[f'{v:.2f}' for v in vs]} are "
                        f"not strictly decreasing; the paragraph says the index "
                        f"decreases with N on all three examples")
    return msgs


def compute_p2_apost_caption() -> dict:
    """Part II fig:stage1-bounds caption: Ex2/3 vs Ex1 measured-error spread
    (percent) over N >= 12500."""
    pct = 0.0
    for N in (12500, 50000):
        base = float(np.mean(_apost_row("banana", N)["resB"]))
        for t in ("even_fold", "conformal_cube"):
            v = float(np.mean(_apost_row(t, N)["resB"]))
            pct = max(pct, 100.0 * abs(v - base) / base)
    return {"pct": pct}


# ----- Supplement SM3: R_H2 kappa-sweep tables (sm:s2reg-empirical) ----------
# tab:dw-kappa-grid + tab:stage2-headline, from cache/sm3_rh2_kappa_bip.json
# (experiments/exp_sm3_rh2_kappa.py).  Per N, mean Example-1 BIP posterior KL
# (r=4, over trials) for each Stage-2 scheme on a shared CAS subspace:
#   dw-kappa-grid : the 8 R_H2 columns, R_H2(kappa) = hermitescaled:c=0,alpha=0,
#                   delta=kappa; the row min is the oracle-best kappa.
#   headline      : no_reg (hermitescaled delta=1e-12), R_H2 at the row-min
#                   kappa, and proposed (theoretical:c=3,delta=1e-5).
# Bold placement (the per-row argmin) is set in the .tex; this tool syncs the
# numeric values, not which cell is bold, so a moved argmin needs a manual
# re-bold (the audit still catches every value).
_SM3_KGRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
_sm3_cache = None


def _sm3_load() -> dict:
    global _sm3_cache
    if _sm3_cache is None:
        _sm3_cache = _jload("sm3_rh2_kappa_bip.json")
    return _sm3_cache


def _sm3_mean(a) -> float:
    a = np.array([v for v in a if v is not None and not np.isnan(v)])
    return float(a.mean()) if len(a) else float("nan")


def _sm3_kgrid_row(N: int) -> dict:
    c = _sm3_load()["cells"][str(N)]["rh2"]
    return {f"g{i}": _sm3_mean(c[f"{k:g}"]) for i, k in enumerate(_SM3_KGRID)}


def _sm3_headline_row(N: int) -> dict:
    cell = _sm3_load()["cells"][str(N)]
    rh = min(_sm3_mean(cell["rh2"][f"{k:g}"]) for k in _SM3_KGRID)
    return {"nr": _sm3_mean(cell["no_reg"]), "rh": rh,
            "pr": _sm3_mean(cell["proposed"])}


# A table cell that may or may not be wrapped in \mathbf{...}; captures the
# number either way (named group filled in via %).
_SM3_CELL = r"\$(?:\\mathbf\{)?(?P<%s>[\d.]+)\}?\$"


# ----- Supplement SM3: Stage-2 c-stability table (tab:s2reg-ablations) -------
# Source: cache/sm3_c_stability_banana.json (experiments/exp_sm3_c_stability.py).
# Example-1 banana test-LL, R at c in {1,3,10} vs R_H2 (kappa=1e-3) on the
# SHARED CAS subspace per (r,N,seed) -> the gap isolates the Stage-2
# regularizer.  Per-cell statistic = mean over seeds of (LL[R,c]-LL[R_H2]);
# the row reports mean/min/max of that per-cell gap over the 15 (r,N) cells
# plus lost (<-0.05) / won (>+0.05) cell counts.  The c=3 row bolds its min
# cell and lost count (the selection criterion: smallest worst-case
# give-back); _SM3_CSTAB_R/N below mirror the generator's grid.
_SM3_CSTAB_R = (2, 3, 4, 6, 8)
_SM3_CSTAB_N = (500, 1000, 2000)


def _sm3_cstab_row(c: int) -> dict:
    d = _jload("sm3_c_stability_banana.json")["cells"]
    gaps = []
    for r in _SM3_CSTAB_R:
        for N in _SM3_CSTAB_N:
            cell = d[str(r)][str(N)]
            g = np.array(cell[f"c{c}"]) - np.array(cell["rh2"])
            gaps.append(float(g.mean()))
    a = np.array(gaps)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "lost": float((a < -0.05).sum()), "won": float((a > 0.05).sum())}


def _sgn3(x: float) -> str:
    """Signed 3-decimal, matching the c-stability table's +0.248 / -0.015."""
    return f"{x:+.3f}"


def guard_sm3_cstab_c3(before: dict, after: dict) -> list[str]:
    # Prose adopts c=3 as the nominal: no lost cell, beats R_H2 on average,
    # smallest worst-case give-back of the three.
    msgs = []
    if after["lost"] > 0:
        msgs.append(f"C=3 now loses {after['lost']:.0f} cell(s); prose adopts C=3 "
                    f"as the no-loss nominal with the smallest worst-case give-back")
    if after["mean"] <= 0:
        msgs.append(f"C=3 mean gap now {after['mean']:+.3f} (<=0); prose says all "
                    f"three c beat R_H2 on average")
    return msgs


# ----- Supplement SM3: centering constraint (tab:sm2-centering + -k2) -------
# Source: cache/v83_centering_ablation_{bench}.json and
# cache/v83_k2_sweep_centering_{bench}.json (experiments/exp_centering_*.py).
# Each cell is one (N, seed) or (K_2, seed) realization holding subspace, rank,
# marginals and regularizer fixed, so dkl_constrained_minus_unconstrained is a
# paired gap isolating the constraint.
_CTR_BENCH = {"e1": "banana", "e2": "even_fold", "e3": "conformal_cube"}
_CTR_NS = (500, 2500, 12500, 50000)
_CTR_K2 = {3: 30, 4: 64, 5: 116, 6: 190}
_ctr_cache: dict = {}


def _ctr_load(kind: str, bench: str) -> list:
    k = (kind, bench)
    if k not in _ctr_cache:
        _ctr_cache[k] = _jload(f"v83_{kind}_{bench}.json")["cells"]
    return _ctr_cache[k]


def _ctr_stats(cells: list, field: str = "dkl_constrained_minus_unconstrained"):
    a = np.array([c[field] for c in cells], dtype=float)
    return float(a.mean()), float(a.std())


def _centering_row(N: int) -> dict:
    """One N-row of tab:sm2-centering: the paired gap on each example as
    mean/sd over realizations, plus the Example-1 unconstrained centering
    residual whose failure to reach zero is the point of the paragraph."""
    out = {}
    for g, bench in _CTR_BENCH.items():
        cells = [c for c in _ctr_load("centering_ablation", bench) if c["N"] == N]
        out[f"{g}m"], out[f"{g}s"] = _ctr_stats(cells)
    ex1 = [c for c in _ctr_load("centering_ablation", "banana") if c["N"] == N]
    out["ru"], _ = _ctr_stats(ex1, "centering_residual_unconstrained")
    return out


def _centering_k2_row(K: int) -> dict:
    out = {}
    for g, bench in _CTR_BENCH.items():
        cells = [c for c in _ctr_load("k2_sweep_centering", bench)
                 if c["K_inner"] == K]
        out[f"{g}m"], out[f"{g}s"] = _ctr_stats(cells)
    return out


def _centering_resid() -> dict:
    """The In-sample identity paragraph: the Example-1 and Example-3 residual
    at the two ends of the scan, and the Example-2 range across all N."""
    def ru(bench: str, N: int) -> float:
        cells = [c for c in _ctr_load("centering_ablation", bench) if c["N"] == N]
        return _ctr_stats(cells, "centering_residual_unconstrained")[0]
    e2 = [ru("even_fold", N) for N in _CTR_NS]
    return {"r1lo": ru("banana", 500), "r1hi": ru("banana", 50000),
            "r3lo": ru("conformal_cube", 500), "r3hi": ru("conformal_cube", 50000),
            "r2lo": min(e2), "r2hi": max(e2)}


def _sgn4(x: float) -> str:
    """Signed 4-decimal, matching the centering tables' +0.0250 / -0.0046."""
    return f"{x:+.4f}"


def _ctr_cell(g: str) -> str:
    return r"\$(?P<" + g + r"m>[+-][\d.]+)\{\\pm\}(?P<" + g + r"s>[\d.]+)\$"


def guard_centering_row(before: dict, after: dict) -> list[str]:
    # Prose: "The constrained estimator attains the lower divergence at every
    # sample size on all three examples."  Any nonnegative mean breaks it.
    bad = [g for g in ("e1", "e2", "e3") if after[f"{g}m"] >= 0]
    if bad:
        return [f"centering gap is no longer negative on {', '.join(bad)}; the "
                f"paragraph says the constrained solve attains the lower "
                f"divergence at every sample size on all three examples"]
    return []


def guard_centering_k2(before: dict, after: dict, K: int = 0) -> list[str]:
    # Prose: at K2=3 the constraint costs about +0.05 nats on Examples 1-2,
    # two orders of magnitude above tab:sm2-centering; at K2>=4 it is at most
    # 0.006 nats on those two.
    msgs = []
    if K == 3:
        for g, ex in (("e1", "1"), ("e2", "2")):
            if after[f"{g}m"] < 0.02:
                msgs.append(f"K2=3 cost on Example~{ex} is now {after[f'{g}m']:+.4f} "
                            f"(<0.02); the paragraph says two orders of magnitude "
                            f"above the gaps of tab:sm2-centering")
    else:
        for g, ex in (("e1", "1"), ("e2", "2")):
            if abs(after[f"{g}m"]) > 0.006:
                msgs.append(f"K2={K} gap on Example~{ex} is now {after[f'{g}m']:+.4f} "
                            f"(>0.006 in magnitude); the paragraph bounds every "
                            f"K2>=4 cell on Examples 1-2 by 0.006 nats")
    return msgs


# --------------------------------------------------------------------------
# Narrative guards.  Return a list of human-readable warnings when a value
# change may invalidate surrounding prose.  Empty list == safe to apply.
# --------------------------------------------------------------------------
def guard_spectrum(before: dict, after: dict) -> list[str]:
    msgs = []
    if sorted(("e1", "e2", "e3", "e4"), key=lambda g: -after[g]) != ["e1", "e2", "e3", "e4"]:
        msgs.append("top-4 eigenvalues are no longer in descending order as written")
    # The prose asserts a CLIFF after lambda_b (b = B_SUPPORT = the mixing
    # support), and that the copula rank is b rather than the latent rank 4.  A
    # graded tail here would mean the linear score term (I - Sigma^-1)z is back,
    # i.e. the mixing is no longer orthogonal -- which is precisely the artifact
    # the redesign removed, and which the old prose described.
    cliff = after.get("_cliff")
    if cliff is not None and cliff < 5.0:
        msgs.append(
            f"lambda_{B_SUPPORT}/lambda_{B_SUPPORT + 1} = {cliff:.1f} (<5): the spectrum is "
            f"graded, not a cliff.  The prose of sec:exp-benchmark asserts a cliff after "
            f"eigenvalue {B_SUPPORT} and attributes it to the orthogonal mixing; check that "
            f"Sigma = I still holds (a non-orthogonal mixing reintroduces the full-rank "
            f"linear term and regrades the tail)"
        )
    return msgs


def guard_floor(before: dict, after: dict) -> list[str]:
    """Defends the non-numeric claims of the reworked §5.1 paragraph on
    every run2026 subspace cell."""
    msgs = []
    data = _sec51_run2026()
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    qlow = int(after["qlow"])
    for tag, per_n in data.items():
        Ns = sorted(per_n)
        cas = {N: mean(per_n[N]["cas"]) for N in Ns}
        # "The CAS angle decreases with N": first-to-last decrease, no
        # material increase between consecutive N.
        if cas[Ns[-1]] >= cas[Ns[0]]:
            msgs.append(f"{tag}: CAS mean sinTheta does not decrease over the scan")
        for a, b in zip(Ns, Ns[1:]):
            if cas[b] > 1.10 * cas[a]:
                msgs.append(f"{tag}: CAS mean sinTheta increases "
                            f"{cas[a]:.2f} -> {cas[b]:.2f} at N={b}")
        # "the angle decreases by an order of magnitude" when the selection
        # moves: q1==qlow seeds at N >= 10000 vs the headline mean.
        plateau = [v for N in Ns if N >= 10000
                   for s, v in per_n[N]["cas"].items()
                   if per_n[N]["kq"][s][0] == qlow]
        if plateau and np.mean(plateau) / cas[Ns[-1]] < 5.0:
            msgs.append(f"{tag}: crossing lowers the angle by only "
                        f"{np.mean(plateau)/cas[Ns[-1]]:.1f}x; prose says an "
                        f"order of magnitude")
        # "angles of PCA and of a uniformly random subspace are N-independent"
        for name in ("pca", "rand"):
            m = [mean(per_n[N][name]) for N in Ns]
            if (max(m) - min(m)) / float(np.mean(m)) > 0.2:
                msgs.append(f"{tag}: {name} mean sinTheta varies "
                            f"{min(m):.2f}-{max(m):.2f} across N; prose says "
                            f"N-independent")
    # "In the crossover region both dictionaries appear seed by seed"
    if not any(len({q for (q, _e) in per_n[N]["kq"].values()}) > 1
               for per_n in data.values() for N in per_n):
        msgs.append("no cell shows a mixed per-seed selection; the crossover "
                    "sentence breaks")
    return msgs


def _rank_ok(after: dict) -> bool:
    return after["cas"] < min(after["pca"], after["pom"], after["gauss"])


def guard_section52(before: dict, after: dict) -> list[str]:
    msgs = []
    if not _rank_ok(after):
        msgs.append("CAS-HCSM is no longer the lowest noise KL -- the §5.2 ranking prose breaks")
    return msgs


def guard_section53(before: dict, after: dict) -> list[str]:
    msgs = []
    if not _rank_ok(after):
        msgs.append("CAS-HCSM is no longer the lowest posterior KL -- the §5.3 ranking prose breaks")
    return msgs


def guard_jhi_row(before: dict, after: dict, ex: str = "") -> list[str]:
    """Sandwich + bound-above-measured + cross-cache agreement per table row."""
    msgs = []
    if after["jhi"] > after["trace"]:
        msgs.append(f"Example~{ex}: J_hi {after['jhi']:.4f} exceeds the trace "
                    f"bound {after['trace']:.4f}; eq:sm-jhi-sandwich breaks")
    if after["dm"] + 4 * after["ds"] > after["jhi"]:
        msgs.append(f"Example~{ex}: measured divergence {after['dm']:.4f} (+4 SE) "
                    f"reaches J_hi {after['jhi']:.4f}; the bound no longer holds "
                    f"-- check the quadrature cells for outlier u's before "
                    f"applying")
    label = {"1": "example1", "2": "example2", "3": "example3"}[ex]
    jt = float(_jhi_cache()[label]["J_trace_at_Vtrace"])
    if abs(jt - after["trace"]) / after["trace"] > 0.02:
        msgs.append(f"Example~{ex}: the jhi-cache trace {jt:.4f} and the "
                    f"stage1_cert rank_price {after['trace']:.4f} disagree by "
                    f">2%; the two caches were produced under different "
                    f"configurations -- diagnose before applying")
    return msgs


def guard_jhi_reading(before: dict, after: dict) -> list[str]:
    msgs = []
    for ex in "123":
        r = _jhi_row(f"example{ex}")
        # bound above measured on each example
        if r["dm"] + 4 * r["ds"] > r["jhi"]:
            msgs.append(f"Example~{ex}: measured divergence reaches J_hi; see "
                        f"the jhi_row guard")
    return msgs


# --------------------------------------------------------------------------
# Claim registry
# --------------------------------------------------------------------------
@dataclass
class Claim:
    key: str
    file: str
    section: str
    pattern: str
    fmt: dict                      # group_name -> python format spec, e.g. "{:.2f}"
    compute: Callable[[], dict]    # -> {group_name: float}
    narrative_guard: Optional[Callable[[dict, dict], list]] = None
    flags: int = 0                 # re flags
    rel_note: float = 0.15         # soft "large change" threshold for the audit note
    occurrence: Optional[int] = None  # if the pattern repeats (same row label in
                                   # several table blocks), pick this 0-based match
                                   # instead of erroring on non-uniqueness


REGISTRY: list[Claim] = [
    Claim(
        key="spectrum",
        file=MAIN,
        section="§5.1 Example-1 score-covariance spectrum",
        pattern=(
            r"top-\$4\$ eigenvalues \$\\approx (?P<e1>[\d.]+), (?P<e2>[\d.]+), "
            r"(?P<e3>[\d.]+), (?P<e4>[\d.]+)\$, with the remaining \$16\$ eigenvalues summing to \$\\approx (?P<tail>[\d.]+)\$"
        ),
        fmt={"e1": "{:.2f}", "e2": "{:.2f}", "e3": "{:.2f}", "e4": "{:.2f}", "tail": "{:.2f}"},
        compute=compute_spectrum,
        narrative_guard=guard_spectrum,
    ),
    Claim(
        key="spectrum_ex2",
        file=MAIN,
        section="§5.1 Example-2 score-covariance spectrum",
        pattern=(
            r"Its top-\$4\$ score-covariance eigenvalues are \$\\approx (?P<e1>[\d.]+), "
            r"(?P<e2>[\d.]+), (?P<e3>[\d.]+), (?P<e4>[\d.]+)\$, with the remaining \$16\$ "
            r"summing to \$\\approx (?P<tail>[\d.]+)\$"
        ),
        fmt={"e1": "{:.2f}", "e2": "{:.2f}", "e3": "{:.2f}", "e4": "{:.2f}", "tail": "{:.2f}"},
        compute=compute_spectrum_ex2,
        narrative_guard=guard_spectrum,
        occurrence=0,
    ),
    Claim(
        key="spectrum_ex3",
        file=MAIN,
        section="§5.1 Example-3 score-covariance spectrum",
        pattern=(
            r"Its top-\$4\$ score-covariance eigenvalues are \$\\approx (?P<e1>[\d.]+), "
            r"(?P<e2>[\d.]+), (?P<e3>[\d.]+), (?P<e4>[\d.]+)\$, with the remaining \$16\$ "
            r"summing to \$\\approx (?P<tail>[\d.]+)\$"
        ),
        fmt={"e1": "{:.2f}", "e2": "{:.2f}", "e3": "{:.2f}", "e4": "{:.2f}", "tail": "{:.2f}"},
        compute=compute_spectrum_ex3,
        narrative_guard=guard_spectrum,
        occurrence=1,
    ),
    Claim(
        key="sec51_floor",
        file=MAIN,
        section="§5.1 subspace recovery (seeds, selected interaction orders, crossover story)",
        pattern=(
            r"over \$(?P<seeds>\d+)\$ independent realizations at each \$N\$, with the multi-index set of each estimate "
            r"chosen by cross-validation on its own sample.*?"
            r"the cross-validation criterion chooses interaction order \$q_1 = (?P<qlow>\d+)\$.*?"
            r"a higher interaction order \(\$q_1 = (?P<qhi12>\d+)\$ on Examples~1--2, "
            r"\$q_1 = (?P<qhi3>\d+)\$ with even degrees on Example~3\)"
        ),
        fmt={"seeds": "{:.0f}", "qlow": "{:.0f}", "qhi12": "{:.0f}", "qhi3": "{:.0f}"},
        compute=compute_floor,
        narrative_guard=guard_floor,
        flags=re.DOTALL,
    ),
    Claim(
        key="sec52",
        file=MAIN,
        section="§5.2 noise KL (CAS/PCA/PoM/Gaussian)",
        pattern=(
            r"mean \$(?P<cas>[\d.]+)\$; \\Cref\{fig:testll\}\), versus \$(?P<pca>[\d.]+)\$ "
            r"for PCA-HCSM, \$(?P<pom>[\d.]+)\$ for PoM, and \$(?P<gauss>[\d.]+)\$ for the "
            r"Gaussian copula"
        ),
        fmt={"cas": "{:.3f}", "pca": "{:.2f}", "pom": "{:.2f}", "gauss": "{:.2f}"},
        compute=compute_section52,
        narrative_guard=guard_section52,
    ),
    Claim(
        key="sec53",
        file=MAIN,
        section="§5.3 BIP posterior KL (CAS/PCA/PoM/Gaussian)",
        pattern=(
            r"mean posterior KL divergence \$(?P<cas>[\d.]+)\$, versus \$(?P<pca>[\d.]+)\$ for PCA-HCSM, "
            r"\$(?P<pom>[\d.]+)\$ for PoM, and \$(?P<gauss>[\d.]+)\$ for the Gaussian-copula baseline"
        ),
        fmt={"cas": "{:.3f}", "pca": "{:.2f}", "pom": "{:.2f}", "gauss": "{:.2f}"},
        compute=compute_section53,
        narrative_guard=guard_section53,
    ),
    # ----- supplement: tab:sm-jhi rows + Interpretation (three-source) -------
    Claim(
        key="jhi_row1",
        file=P2SUPP,
        section="SM tab:sm-jhi row Example 1 (trace bound, J_hi, psi-gap, DKL, sinTheta)",
        pattern=(
            r"Example~1\s*&\s*\$(?P<trace>[\d.]+)\$\s*&\s*\$(?P<jhi>[\d.]+)\$\s*&\s*"
            r"\$(?P<gm>[\d.]+)\\cdot10\^\{-(?P<ge>\d+)\}\$\s*&\s*"
            r"\$(?P<dm>[\d.]+)\{\\pm\}(?P<ds>[\d.]+)\$\s*&\s*\$(?P<st>[\d.]+)\$"
        ),
        fmt={"trace": "{:.4f}", "jhi": "{:.4f}", "gm": "{:.1f}", "ge": "{:.0f}",
             "dm": "{:.4f}", "ds": "{:.4f}", "st": "{:.3f}"},
        compute=compute_jhi_row1,
        narrative_guard=(lambda b, a: guard_jhi_row(b, a, "1")),
    ),
    Claim(
        key="jhi_row2",
        file=P2SUPP,
        section="SM tab:sm-jhi row Example 2",
        pattern=(
            r"Example~2\s*&\s*\$(?P<trace>[\d.]+)\$\s*&\s*\$(?P<jhi>[\d.]+)\$\s*&\s*"
            r"\$(?P<gm>[\d.]+)\\cdot10\^\{-(?P<ge>\d+)\}\$\s*&\s*"
            r"\$(?P<dm>[\d.]+)\{\\pm\}(?P<ds>[\d.]+)\$\s*&\s*\$(?P<st>[\d.]+)\$"
        ),
        fmt={"trace": "{:.4f}", "jhi": "{:.4f}", "gm": "{:.1f}", "ge": "{:.0f}",
             "dm": "{:.4f}", "ds": "{:.4f}", "st": "{:.3f}"},
        compute=compute_jhi_row2,
        narrative_guard=(lambda b, a: guard_jhi_row(b, a, "2")),
    ),
    Claim(
        key="jhi_row3",
        file=P2SUPP,
        section="SM tab:sm-jhi row Example 3",
        pattern=(
            r"Example~3\s*&\s*\$(?P<trace>[\d.]+)\$\s*&\s*\$(?P<jhi>[\d.]+)\$\s*&\s*"
            r"\$(?P<gm>[\d.]+)\\cdot10\^\{-(?P<ge>\d+)\}\$\s*&\s*"
            r"\$(?P<dm>[\d.]+)\{\\pm\}(?P<ds>[\d.]+)\$\s*&\s*\$(?P<st>[\d.]+)\$"
        ),
        fmt={"trace": "{:.4f}", "jhi": "{:.4f}", "gm": "{:.1f}", "ge": "{:.0f}",
             "dm": "{:.4f}", "ds": "{:.4f}", "st": "{:.3f}"},
        compute=compute_jhi_row3,
        narrative_guard=(lambda b, a: guard_jhi_row(b, a, "3")),
    ),
    Claim(
        key="jhi_reading",
        file=P2SUPP,
        section="SM sharp-bound Interpretation (gap pct, H_perp range/departure, factors x3, sinTheta)",
        pattern=(
            r"The gap \\eqref\{eq:sm-jhi-trace-gap\} is below \$(?P<pct>[\d.]+)\\%\$ "
            r"of the trace bound on each"
            r".*?lies in \$\[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\]\$"
            r".*?differ from \$1\$ by at most \$(?P<dep>[\d.]+)\$"
            r".*?exceeds the divergence by factors of \$(?P<f1>[\d.]+)\$, \$(?P<f2>[\d.]+)\$ "
            r"and \$(?P<f3>[\d.]+)\$, and \$J_\{\\mathrm\{hi\}\}\$ by "
            r"\$(?P<j1>[\d.]+)\$, \$(?P<j2>[\d.]+)\$ and \$(?P<j3>[\d.]+)\$"
            r".*?lies within \$\\\|\\sin\\Theta\\\|_F \\le (?P<st>[\d.]+)\$"
        ),
        fmt={"pct": "{:.1f}", "lo": "{:.3f}", "hi": "{:.3f}", "dep": "{:.3f}",
             "f1": "{:.2f}", "f2": "{:.2f}", "f3": "{:.2f}",
             "j1": "{:.2f}", "j2": "{:.2f}", "j3": "{:.2f}", "st": "{:.3f}"},
        compute=compute_jhi_reading,
        narrative_guard=guard_jhi_reading,
        flags=re.DOTALL,
    ),
]


# ----- SM1 table-row claims, built in a loop and appended to REGISTRY --------
for _r in (2, 3, 4, 6, 8):
    _cells, _fmt = [], {}
    for _N in _SM1_NS:
        # bold is cosmetic (it marks the >=0.05-nat winner per the caption);
        # tolerate its presence or absence so the r=2 near-tie row (no bold)
        # still matches the row claim.
        _cells.append(rf"\$(?:\\mathbf\{{)?(?P<cas{_N}>[-\d.]+)\}}?\$")
        _cells.append(rf"\$(?:\\mathbf\{{)?(?P<pca{_N}>[-\d.]+)\}}?\$")
        _fmt[f"cas{_N}"] = "{:.2f}"
        _fmt[f"pca{_N}"] = "{:.2f}"
    REGISTRY.append(Claim(
        key=f"sm1_banana_r{_r}", file=SUPP,
        section=f"SM tab:pca-banana row r={_r} (CAS/PCA test LL at N=500/1000/2000)",
        pattern=rf"\n{_r}  & " + " & ".join(_cells) + r" \\\\",
        fmt=_fmt,
        compute=(lambda rr=_r: _sm1_banana_row(rr)),
        narrative_guard=(lambda b, a, rr=_r: guard_sm1_banana(b, a, rr)),
    ))

_SM1_BIP_ROWS = [
    ("Gaussian", r"Gaussian\s+&", False, False),
    ("PoM", r"Product of marginals\s+&", True, False),
    ("PCA r=2", r"PCA, \$r=2\$\s+&", True, False),
    ("PCA r=3", r"PCA, \$r=3\$\s+&", True, False),
    ("PCA r=4", r"PCA, \$r=4\$\s+&", True, False),
    ("PCA r=6", r"PCA, \$r=6\$\s+&", True, False),
    ("PCA r=8", r"PCA, \$r=8\$\s+&", True, False),
    ("CAS r=4", r"\\textsc\{Cas\}, \$r=4\$\s+&", True, True),
]
for _name, _prefix, _has_red, _bold in _SM1_BIP_ROWS:
    def _bcell(g, bold):
        return (rf"\$\\mathbf\{{(?P<{g}>[-\d.]+)\}}\$" if bold
                else rf"\$(?P<{g}>[-\d.]+)\$")
    _pat = _prefix + " " + " & ".join(_bcell(g, _bold) for g in ("mean", "median", "worst"))
    _fmt = {g: "{:.2f}" for g in ("mean", "median", "worst")}
    if _has_red:
        _pat += " & " + (rf"\$\\mathbf\{{(?P<red>[-\d]+)\\%\}}\$" if _bold
                         else rf"\$(?P<red>[-\d]+)\\%\$")
        _fmt["red"] = _pct
    else:
        _pat += r" & n/a"
    _pat += r" \\\\"
    REGISTRY.append(Claim(
        key=f"sm1_bip_{_name.replace(' ', '_').replace('=', '')}",
        file=SUPP, section=f"SM tab:pca-bip row {_name}",
        pattern=_pat, fmt=_fmt,
        compute=(lambda nm=_name: _sm1_bip_row(nm)),
        narrative_guard=(guard_sm1_bip_cas if _name == "CAS r=4" else None),
    ))


# ----- Part II sec:tightness prose claims ----------------------------------
REGISTRY.append(Claim(
    key="cor44_slack_looseness", file=P2MAIN,
    section="Part II sec:tightness Looseness (bound-to-divergence ratio x3)",
    # The divergences, standard errors and trace bounds themselves are printed
    # in the supplement table tab:sm-jhi (claims jhi_row1-3); the main text
    # prints only the ratios.  The guard still checks bound > divergence.
    pattern=(r"The ratio of bound to divergence equals "
             r"\$(?P<f1>[\d.]+)\$, \$(?P<f2>[\d.]+)\$ and \$(?P<f3>[\d.]+)\$ "
             r"on Examples~1, 2 and~3"),
    fmt={"f1": "{:.2f}", "f2": "{:.2f}", "f3": "{:.2f}"},
    compute=compute_cor44_looseness,
    narrative_guard=guard_cor44_looseness,
))
REGISTRY.append(Claim(
    key="cor44_slack_ratio_range", file=P2MAIN,
    section="Part II sec:tightness ratio at the estimated subspace (Ex1, N>=2500)",
    pattern=(r"lies between \$(?P<rlo>[\d.]+)\$ and \$(?P<rhi>[\d.]+)\$ on "
             r"Example~1 for every \$N \\ge 2\{,\}500\$"),
    fmt={"rlo": "{:.2f}", "rhi": "{:.2f}"},
    compute=compute_cor44_ratio_range,
    narrative_guard=guard_cor44_ratio_range,
))
REGISTRY.append(Claim(
    key="p2_tight_jhipct", file=P2MAIN,
    section="Part II sec:tightness J_hi-below-trace percentage (two-subspaces paragraph)",
    pattern=(r"is less than \$(?P<pct>[\d.]+)\\%\$ below the bound "
             r"\\eqref\{I-eq:kl-trunc\} on each problem"),
    fmt={"pct": "{:.1f}"},
    compute=compute_p2_tight_jhipct,
))

# The log-Sobolev looseness range and the sinTheta tolerance are each printed
# at a hooked source and repeated in prose; these claims pin the repeats to
# the same caches (compute_p2_lsi_repeats).
REGISTRY.append(Claim(
    key="p2_lsi_contrib", file=P2MAIN,
    section="Part II contribution 3 log-Sobolev looseness range + sinTheta tolerance",
    pattern=(r"by a factor between \$(?P<lo>[\d.]+)\$ and \$(?P<hi>[\d.]+)\$ on the "
             r"examples, and the minimizer of the tighter bound.*?"
             r"\$\\\|\\sin\\Theta\\\|_F \\le (?P<st>[\d.]+)\$ of the top-\$r\$"),
    fmt={"lo": "{:.2f}", "hi": "{:.2f}", "st": "{:.3f}"},
    compute=compute_p2_lsi_repeats,
    flags=re.DOTALL,
))
REGISTRY.append(Claim(
    key="p2_lsi_tightness_st", file=P2MAIN,
    section="Part II sec:tightness sinTheta tolerance (J_hi minimizer vs V_r^C)",
    pattern=(r"[Tt]he minimizer of \$J_\{\\mathrm\{hi\}\}\$ lies within "
             r"\$\\\|\\sin\\Theta\\\|_F \\le (?P<st>[\d.]+)\$"),
    fmt={"st": "{:.3f}"},
    compute=compute_p2_lsi_repeats,
))
REGISTRY.append(Claim(
    key="p2_lsi_conclusion", file=P2MAIN,
    section="Part II conclusion log-Sobolev looseness range + sinTheta tolerance",
    pattern=(r"by a factor between \$(?P<lo>[\d.]+)\$ and \$(?P<hi>[\d.]+)\$ on the "
             r"three examples, and on each of them.*?"
             r"\$\\\|\\sin\\Theta\\\|_F \\le (?P<st>[\d.]+)\$"),
    fmt={"lo": "{:.2f}", "hi": "{:.2f}", "st": "{:.3f}"},
    compute=compute_p2_lsi_repeats,
    flags=re.DOTALL,
))


# ----- Part II Stage-1 a posteriori inline claims (cache/aposteriori_*.json) --


def compute_p2_hier_trunc() -> dict:
    r"""Part II sec:exp-aposteriori hier-trunc paragraph: the degree-enrichment
    beta floor and the floor/ceil bracket of the three interaction-enrichment
    betas. Sources: cache/hier_trunc_{tag}_q.json (interaction run) and
    cache/hier_trunc_{tag}_K.json (degree run), both written by
    regen-hier-trunc; the file name carries the direction."""
    tags = ("banana", "even_fold", "conformal_cube")

    def _load(name):
        with open(os.path.join(CACHE, name)) as fh:
            return json.load(fh)

    q = [_load(f"hier_trunc_{t}_q.json") for t in tags]
    deg = [_load(f"hier_trunc_{t}_K.json") for t in tags]
    betas = [c["beta_oracle"] for c in q]
    return {"kbmin": min(c["beta_oracle"] for c in deg),
            "blo": math.floor(1000.0 * min(betas)) / 1000.0,
            "bhi": math.ceil(1000.0 * max(betas)) / 1000.0}


REGISTRY.append(Claim(
    key="p2_hier_trunc", file=P2MAIN,
    section=("Part II \u00a76.4 hier-trunc (degree-beta floor, interaction-beta "
             "bracket)"),
    pattern=(r"When the degree is raised, \$\\beta\$ exceeds \$(?P<kbmin>[\d.]+)\$ "
             r"on every example.*?when the interaction order is raised, \$\\beta\$ "
             r"lies between \$(?P<blo>[\d.]+)\$ and \$(?P<bhi>[\d.]+)\$"),
    fmt={"kbmin": "{:.2f}", "blo": "{:.2f}", "bhi": "{:.2f}"},
    compute=compute_p2_hier_trunc,
    flags=re.DOTALL,
))


def compute_p2_proj_threshold() -> dict:
    """Smallest N of the Part II decomposition from which the subspace term is
    the largest of the three reference terms at every larger N, on every example
    (out/<example>_production/exp_stage2_oracle_kl.json)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thr = 0
    for b in ("banana", "even_fold", "conformal_cube"):
        rows = json.load(open(os.path.join(root, "out", f"{b}_production",
                                           "exp_stage2_oracle_kl.json")))["per_N"]
        rows = sorted(rows, key=lambda e: int(e["N"]))
        first = None
        for e in rows:
            largest = e["delta_sub_mean"] >= max(e["delta_marg_mean"], e["delta_den_mean"])
            first = (first or int(e["N"])) if largest else None
        if first is None:
            raise ValueError(f"{b}: subspace term not largest at the largest N")
        thr = max(thr, first)
    return {"nthr": float(thr)}


for _key, _where, _pat in (
    ("p2_proj_thr_intro", "Part II contribution 1 subspace-term threshold",
     r"\$\\mathcal E_\{\\rm proj\}\$ is the largest term for \$N \\ge (?P<nthr>[\d{},]+)\$"),
    ("p2_proj_thr_results", "Part II sec:experiments subspace-term threshold",
     r"[Ff]or \$N \\ge (?P<nthr>[\d{},]+)\$ the subspace term is the largest of the three"),
    ("p2_proj_thr_concl", "Part II conclusion subspace-term threshold",
     r"For \$N \\ge (?P<nthr>[\d{},]+)\$, on all three examples, the largest term"),
):
    REGISTRY.append(Claim(key=_key, file=P2MAIN, section=_where, pattern=_pat,
                          fmt={"nthr": lambda v: f"{v:,.0f}".replace(",", "{,}")},
                          compute=compute_p2_proj_threshold))
REGISTRY.append(Claim(
    key="p2_apost_caption", file=P2MAIN,
    section="Part II fig:stage1-bounds caption (Ex2/3 vs Ex1 spread, percent)",
    pattern=(r"differ from Example~1 by at most \$(?P<pct>[\d.]+)\\%\$ "
             r"for \$N \\ge 12500\$"),
    fmt={"pct": lambda v: f"{math.ceil(v)}"},   # an 'at most' claim: round up
    compute=compute_p2_apost_caption,
))
REGISTRY.append(Claim(
    key="p2_apriori_loose", file=P2MAIN,
    section="Part II \u00a76.4 a priori bound-to-reference factor bracket at N=5e4",
    pattern=(r"exceeds the reference \$\\mathcal E_\{\\rm proj\}\$ by a factor "
             r"between \$(?P<rlo>[\d.]+)\$ and \$(?P<rhi>[\d.]+)\$ across the "
             r"examples at \$N = 5\\times10\^4\$"),
    fmt={"rlo": "{:.1f}", "rhi": "{:.1f}"},
    compute=compute_p2_apriori_loose,
    narrative_guard=guard_p2_apriori_loose,
))
REGISTRY.append(Claim(
    key="p2_e2e_eff", file=P2MAIN,
    section="Part II \u00a76.5 exception effectivity index (Example 3 at largest N) + relations guard",
    pattern=(r"exception is Example~3 at the largest \$N\$, where the "
             r"index is \$(?P<e34>[\d.]+)\$"),
    fmt={"e34": "{:.1f}"},
    compute=compute_p2_e2e_eff,
    narrative_guard=guard_p2_e2e_eff,
))


def compute_p2_apost_models() -> dict:
    r"""Part II sec:exp-aposteriori and sec:limitations: the number of
    estimated models (examples x sample sizes x seeds) in which the bound of
    cor:aposteriori is evaluated; guard_p2_apost_models defends 'lies above
    ... for every seed and every N'."""
    return {"n": float(sum(len(r["rhs"]) for t in _APOST_TAGS
                           for r in _apost(t)["rows"]))}


def guard_p2_apost_models(before: dict, after: dict) -> list[str]:
    miss = [(t, r["N"]) for t in _APOST_TAGS for r in _apost(t)["rows"]
            if min(r["covered"]) < 1.0]
    return ([f"the bound of cor:aposteriori falls below the quantity it bounds "
             f"at {miss}; the text says it lies above it in every model"]
            if miss else [])


for _key, _pat in (
        ("p2_apost_models", r"in all \$(?P<n>\d+)\$ estimated models, and the "
                            r"finite-sample term decreases"),
        ("p2_lim_models", r"exceeds the trace in all \$(?P<n>\d+)\$ "
                          r"estimated models")):
    REGISTRY.append(Claim(
        key=_key, file=P2MAIN,
        section="Part II number of estimated models of cor:aposteriori, all covered",
        pattern=_pat, fmt={"n": "{:.0f}"},
        compute=compute_p2_apost_models,
        narrative_guard=guard_p2_apost_models,
    ))


def _e2e_rows() -> list:
    return [r for t in _APOST_TAGS for r in _jload(f"apost_e2e_{t}.json")]


def compute_p2_e2e_models() -> dict:
    r"""Part II sec:exp-endtoend: the number of estimated models of the
    end-to-end comparison; guard_p2_e2e_models defends the two sentences that
    follow it (marginal estimate within twice the combined standard errors in
    every model, density estimate biased upward by about one combined standard
    error at every N)."""
    return {"n": float(len(_e2e_rows()))}


def guard_p2_e2e_models(before: dict, after: dict) -> list[str]:
    msgs = []
    rows = _e2e_rows()
    out = [(r["benchmark"], r["N"], r["seed"]) for r in rows
           if abs(r["dhat_marg"] + r["log_mass"] - r["ref_delta_marg"])
           > 2.0 * float(np.hypot(r["se_marg"], r["ref_delta_marg_se"]))]
    if out:
        msgs.append(f"marginal estimate outside twice the combined standard "
                    f"errors at {out}; the text says 'in all' models")
    for t in _APOST_TAGS:
        e2e = _jload(f"apost_e2e_{t}.json")
        for N in sorted({r["N"] for r in e2e}):
            cs = [r for r in e2e if r["N"] == N]
            bias = float(np.mean([r["dhat_den"] - r["ref_delta_den"] for r in cs]))
            se = float(np.mean([np.hypot(r["se_den"], r["ref_delta_den_se"])
                                for r in cs]))
            if not 0.5 <= bias / se <= 1.5:
                msgs.append(f"{t} N={N}: density-estimate bias is {bias / se:.2f} "
                            f"combined standard errors; the text says 'about one "
                            f"combined standard error at every N'")
    return msgs


REGISTRY.append(Claim(
    key="p2_e2e_models", file=P2MAIN,
    section="Part II sec:exp-endtoend model count, marginal agreement, density bias",
    pattern=r"In all \$(?P<n>\d+)\$ estimated models, \$\\widehat\{\\mathcal "
            r"E\}_\{\\rm marg\}\$ lies within twice",
    fmt={"n": "{:.0f}"},
    compute=compute_p2_e2e_models,
    narrative_guard=guard_p2_e2e_models,
))


def compute_p2_lim_trc() -> dict:
    r"""Part II sec:limitations: on Example 3, the factor by which raising the
    Stage-1 interaction order from (4,2) to (4,3) multiplies tr(C_Lambda), and
    the percentage by which raising the degree to (5,2) increases it, from the
    N_ref = 10^6 multi-index sequence (cor44 saturation cache)."""
    rows = {(r["K"], r["q"]): float(r["trC"]) for r in
            _jload("cor44_conformal_cube_saturation_production_N1000000.json")["rows"]}
    base = rows[(4, 2)]
    return {"q": rows[(4, 3)] / base, "k": 100.0 * (rows[(5, 2)] / base - 1.0)}


def compute_sm_ds_forms() -> dict:
    r"""Part II supplement sm:disintegration-mc: the factor by which the
    standard error of the conditional form of the divergence is smaller than
    that of the signed form, floored and ceiled over the three examples
    (delta_sub_Z_{law}.json, both forms computed in the same run)."""
    rat = []
    for law in _APOST_TAGS:
        d = _jload(f"delta_sub_Z_{law}.json")
        rat.append(d["diff_estimator_se"] / d["delta_sub_Z_se"])
    return {"lo": float(math.floor(min(rat))), "hi": float(math.ceil(max(rat)))}


def guard_sm_ds_forms(before: dict, after: dict) -> list[str]:
    out = []
    for law in _APOST_TAGS:
        d = _jload(f"delta_sub_Z_{law}.json")
        z = abs(d["delta_sub_Z_mean"] - d["diff_estimator_mean"]) / d["diff_estimator_se"]
        if z >= 2.0:
            out.append(f"{law}: the two forms differ by {z:.2f} standard errors of "
                       f"the signed form; the text says less than twice")
    return out


REGISTRY.append(Claim(
    key="sm_ds_forms", file=P2SUPP,
    section="Part II SM conditional Monte Carlo: conditional vs signed form (SE ratio range, agreement guard)",
    pattern=(r"standard error of the conditional form is smaller by factors between "
             r"\$(?P<lo>\d+)\$ and \$(?P<hi>\d+)\$"),
    fmt={"lo": "{:.0f}", "hi": "{:.0f}"},
    compute=compute_sm_ds_forms,
    narrative_guard=guard_sm_ds_forms,
))


REGISTRY.append(Claim(
    key="p2_lim_trc", file=P2MAIN,
    section="Part II sec:limitations tr(C_Lambda) ratio (interaction) and increase (degree), Example 3",
    pattern=(r"Raising the interaction order by one multiplies \$\\tr\(\\mC_\\Lambda\)\$ by "
             r"\$(?P<q>[\d.]+)\$, whereas raising the degree by one increases it by "
             r"\$(?P<k>[\d.]+)\\%\$"),
    fmt={"q": "{:.1f}", "k": "{:.1f}"},
    compute=compute_p2_lim_trc,
))


# ----- SM3 R_H2 kappa-grid + headline row claims (one per N each) -----------
_SM3_NLABELS = {100: r"100", 500: r"500", 2500: r"2\{,\}500",
                12500: r"12\{,\}500", 50000: r"50\{,\}000"}
for _N, _lab in _SM3_NLABELS.items():
    _cells = r"\s*&\s*".join(_SM3_CELL % f"g{i}" for i in range(8))
    REGISTRY.append(Claim(
        key=f"sm3_kgrid_N{_N}", file=SUPP,
        section=f"SM tab:dw-kappa-grid row N={_N} (8 R_H2 kappa columns)",
        pattern=rf"\${_lab}\$\s*&\s*" + _cells,
        fmt={f"g{i}": "{:.2f}" for i in range(8)},
        compute=(lambda NN=_N: _sm3_kgrid_row(NN)),
    ))
for _N, _lab in _SM3_NLABELS.items():
    _pat = (rf"\${_lab}\$\s*&\s*" + (_SM3_CELL % "nr") + r"\s*&\s*"
            + (_SM3_CELL % "rh") + r"\s*&\s*\$[^$]*\^[^$]*\$\s*&\s*"
            + (_SM3_CELL % "pr"))
    REGISTRY.append(Claim(
        key=f"sm3_headline_N{_N}", file=SUPP,
        section=f"SM tab:stage2-headline row N={_N} (no_reg/R_H2-oracle/proposed)",
        pattern=_pat,
        fmt={"nr": "{:.2f}", "rh": "{:.2f}", "pr": "{:.2f}"},
        compute=(lambda NN=_N: _sm3_headline_row(NN)),
    ))


# ----- SM3 Stage-2 c-stability row claims (tab:s2reg-ablations) -------------
# One claim per R c-row; mean/min/max are signed 3dp, lost/won are integer cell
# counts.  The c=3 row bolds its min cell and lost count, so those two cells use
# a \mathbf-wrapped capture; the c=3 row also carries the no-loss/beats-on-
# average narrative guard.  Row labels (\$c=1\$ etc.) are unique to this table.
_SM3_CSTAB_ROWS = [
    (1,  r"\$\\mR\$, \$C=1\$",              False, False, None),
    (3,  r"\$\\mR\$, \$C=3\$ \(proposed\)", True,  True,  guard_sm3_cstab_c3),
    (10, r"\$\\mR\$, \$C=10\$",             False, False, None),
]
for _c, _lab, _minb, _lostb, _guard in _SM3_CSTAB_ROWS:
    _mean = r"\$(?P<mean>[+-][\d.]+)\$"
    _min = (r"\$\\mathbf\{(?P<min>[+-][\d.]+)\}\$" if _minb
            else r"\$(?P<min>[+-][\d.]+)\$")
    _max = r"\$(?P<max>[+-][\d.]+)\$"
    _lost = (r"\$\\mathbf\{(?P<lost>\d+)\}\$" if _lostb else r"(?P<lost>\d+)")
    _won = r"(?P<won>\d+)"
    _pat = (_lab + r"\s*&\s*" + _mean + r"\s*&\s*" + _min + r"\s*&\s*"
            + _max + r"\s*&\s*" + _lost + r"\s*&\s*" + _won)
    REGISTRY.append(Claim(
        key=f"sm3_cstab_c{_c}", file=SUPP,
        section=f"SM tab:s2reg-ablations row R C={_c} (mean/min/max gap, lost/won)",
        pattern=_pat,
        fmt={"mean": _sgn3, "min": _sgn3, "max": _sgn3,
             "lost": "{:.0f}", "won": "{:.0f}"},
        compute=(lambda cc=_c: _sm3_cstab_row(cc)),
        narrative_guard=_guard,
    ))


# ----- Supplement SM3: cross-benchmark regularizer ablation -----------------
# tab:cross-bench-results, from cache/v82_regularizer_ablation_{benchmark}.json
# (banana = Example 1, cubic_banana = Example 2, ppg = Example 3).  Per
# benchmark block:
#   ref-KL line  = mean over seeds of kl_ref_R_th_nominal at each of the 9 N;
#   each Delta row (reg - R_th_nominal) = mean over seeds of
#                  regularizers[reg]['dkl_vs_R_th_nominal'] at each N.
# Label map (table -> cache key): R_th,CV -> R_th_CV; R_cov,id fix ->
#   identity_fixed; R_sob,k fix -> sobolev1_fixed; R_sob,k^2 fix ->
#   sobolev2_fixed; no reg -> no_reg.  (The c/N regularizer flavours are in the
#   cache but omitted from this table, as the caption notes.)  The five row
#   labels repeat once per benchmark block, so each row claim pins its block via
#   `occurrence` (0=Ex1, 1=Ex2, 2=Ex3); the signed +0.00/-0.00 near-zero cells
#   reproduce exactly under the +.2f formatter.
_XB_BENCH = {"banana": 0, "even_fold": 1, "conformal_cube": 2}
_XB_EXNUM = {"banana": "1", "even_fold": "2", "conformal_cube": "3"}
_xb_cache: dict = {}


def _xb_load(bench: str) -> dict:
    if bench not in _xb_cache:
        _xb_cache[bench] = _jload(f"v82_regularizer_ablation_{bench}.json")
    return _xb_cache[bench]


def _xb_mean_per_N(bench: str, fn) -> list:
    d = _xb_load(bench)
    cells = d["cells"]
    out = []
    for N in d["config"]["N_values"]:
        xs = [fn(c) for c in cells if c["N"] == N]
        out.append(float(np.mean(xs)) if xs else float("nan"))
    return out


def _xb_ref(bench: str) -> dict:
    return {f"r{i}": v for i, v in
            enumerate(_xb_mean_per_N(bench, lambda c: c["kl_ref_R_th_nominal"]))}


def _xb_row(bench: str, key: str) -> dict:
    return {f"g{i}": v for i, v in enumerate(_xb_mean_per_N(
        bench, lambda c: c["regularizers"][key]["dkl_vs_R_th_nominal"]))}


def _xb_signed(x: float) -> str:
    return f"{x:+.2f}"


_XB_ROWS = [
    ("R_th_CV",        r"\$\\mR_\{\\rm th,CV\}\$"),
    ("identity_fixed", r"\$\\mR_\{\\rm cov,id\\,fix\}\$"),
    ("sobolev1_fixed", r"\$\\mR_\{\\rm sob,k\\,fix\}\$"),
    ("sobolev2_fixed", r"\$\\mR_\{\\rm sob,k\^2\\,fix\}\$"),
    ("no_reg",         r"\$\\mR_\{\\rm no\\,reg\}\$"),
]
for _bench, _exn in _XB_EXNUM.items():
    _refcells = r",\s*".join(rf"\$(?P<r{i}>[\d.]+)\$" for i in range(9))
    REGISTRY.append(Claim(
        key=f"xbench_ref_{_bench}", file=SUPP,
        section=f"SM tab:cross-bench-results ref-KL line (Example {_exn})",
        pattern=rf"\\textbf\{{Example~{_exn}\}}\s*\\quad\s*\(ref log-score:\s*" + _refcells,
        fmt={f"r{i}": "{:.2f}" for i in range(9)},
        compute=(lambda bb=_bench: _xb_ref(bb)),
    ))
for _key, _lab in _XB_ROWS:
    _rowcells = r"\s*&\s*".join(rf"\$(?P<g{i}>[+-][\d.]+)\$" for i in range(9))
    for _bench, _occ in _XB_BENCH.items():
        REGISTRY.append(Claim(
            key=f"xbench_{_bench}_{_key}", file=SUPP,
            section=f"SM tab:cross-bench-results row {_key} (Example {_XB_EXNUM[_bench]})",
            pattern=_lab + r"\s*&\s*" + _rowcells,
            fmt={f"g{i}": _xb_signed for i in range(9)},
            compute=(lambda bb=_bench, kk=_key: _xb_row(bb, kk)),
            occurrence=_occ,
        ))


# --- SM1 tab:ehat-tightness: accuracy/tightness of eq:Ehat-weyl -----------
# Source: cache/ehat_tightness_{tag}.json (experiments/exp_ehat_tightness.py;
# standalone, no regen cache read). Columns per example row: hatE_r mean +/- sd,
# oracle E_r(C_Lambda), max_s |hatE_r - E_r|, mean sum_{j<=d-r} sigma_j (the
# displayed Lidskii bound), mean (d-r) sigma_1 (its op-norm relaxation), mean
# per-seed ratio of the Lidskii bound to the error.
def _ehat_tight_row(tag: str) -> Callable[[], dict]:
    def compute() -> dict:
        z = _jload(f"ehat_tightness_{tag}.json")
        s, o = z["summary"], z["oracle"]
        return {"ehat": s["Ehat_mean"], "esd": s["Ehat_sd"], "eor": o["E_r"],
                "maxerr": s["abs_err_max"], "bkf": s["bound_kf_mean"],
                "bop": s["bound_op_mean"], "ratio": s["ratio_kf_mean"]}
    return compute


for _i, _tag in enumerate(("banana", "even_fold", "conformal_cube"), start=1):
    REGISTRY.append(Claim(
        key=f"ehat_tight_ex{_i}",
        file=SUPP,
        section=f"SM tab:ehat-tightness row Example {_i} (hatE_r, E_r, max err, KF bound, op bound, ratio)",
        pattern=(
            rf"Example~{_i} & \$(?P<ehat>[\d.]+) \\pm (?P<esd>[\d.]+)\$ & "
            rf"\$(?P<eor>[\d.]+)\$ & \$(?P<maxerr>[\d.]+)\$ & \$(?P<bkf>[\d.]+)\$ & "
            rf"\$(?P<bop>[\d.]+)\$ & \$(?P<ratio>[\d.]+)\$"
        ),
        fmt={"ehat": "{:.4f}", "esd": "{:.4f}", "eor": "{:.4f}",
             "maxerr": "{:.4f}", "bkf": "{:.4f}", "bop": "{:.4f}",
             "ratio": "{:.1f}"},
        compute=_ehat_tight_row(_tag),
    ))


# ----- SM3 centering-constraint claims (tab:sm2-centering, -k2, and the
# In-sample identity paragraph) ---------------------------------------------
_CTR_NLABELS = {500: r"500", 2500: r"2\{,\}500", 12500: r"12\{,\}500",
                50000: r"50\{,\}000"}
for _N, _lab in _CTR_NLABELS.items():
    _cells = r"\s*&\s*".join(_ctr_cell(g) for g in ("e1", "e2", "e3"))
    REGISTRY.append(Claim(
        key=f"ctr_row_N{_N}", file=SUPP,
        section=f"SM tab:sm2-centering row N={_N} (paired gap x3 examples, Ex1 residual)",
        pattern=(rf"\${_lab}\$\s*&\s*" + _cells
                 + r"\s*&\s*\$(?P<ru>[\d.]+)\$"),
        fmt={**{f"{g}m": _sgn4 for g in ("e1", "e2", "e3")},
             **{f"{g}s": "{:.4f}" for g in ("e1", "e2", "e3")},
             "ru": "{:.3f}"},
        compute=(lambda NN=_N: _centering_row(NN)),
        narrative_guard=guard_centering_row,
    ))

for _K, _nb in _CTR_K2.items():
    _cells = r"\s*&\s*".join(_ctr_cell(g) for g in ("e1", "e2", "e3"))
    REGISTRY.append(Claim(
        key=f"ctr_k2_K{_K}", file=SUPP,
        section=f"SM tab:sm2-centering-k2 row K2={_K} (paired gap x3 examples)",
        pattern=(rf"\${_K}\$\s*&\s*\${_nb}\$\s*&\s*" + _cells),
        fmt={**{f"{g}m": _sgn4 for g in ("e1", "e2", "e3")},
             **{f"{g}s": "{:.4f}" for g in ("e1", "e2", "e3")}},
        compute=(lambda KK=_K: _centering_k2_row(KK)),
        narrative_guard=(lambda b, a, KK=_K: guard_centering_k2(b, a, KK)),
    ))

REGISTRY.append(Claim(
    key="ctr_resid", file=SUPP,
    section="SM centering In-sample identity paragraph (unconstrained residual, 3 examples)",
    pattern=(r"Its across-realization mean is \$(?P<r1lo>[\d.]+)\$ at "
             r"\$N = 500\$ and \$(?P<r1hi>[\d.]+)\$ at \$N = 50\{,\}000\$ on "
             r"Example~1, \$(?P<r3lo>[\d.]+)\$ and \$(?P<r3hi>[\d.]+)\$ at those "
             r"sample sizes on Example~3, and lies between \$(?P<r2lo>[\d.]+)\$ and "
             r"\$(?P<r2hi>[\d.]+)\$ at every \$N\$ on Example~2"),
    fmt={g: "{:.3f}" for g in ("r1lo", "r1hi", "r3lo", "r3hi", "r2lo", "r2hi")},
    compute=_centering_resid,
))


# --------------------------------------------------------------------------
# Audit + apply machinery
# --------------------------------------------------------------------------
class ClaimResult:
    def __init__(self, claim: Claim):
        self.claim = claim
        self.error: Optional[str] = None
        self.match: Optional[re.Match] = None
        self.paper: dict = {}      # group -> str (as written)
        self.cache_f: dict = {}    # group -> float (computed)
        self.cache_s: dict = {}    # group -> str (formatted)
        self.stale: dict = {}      # group -> (old_str, new_str)
        self.notes: list[str] = []     # soft large-change notes
        self.narrative: list[str] = [] # narrative-guard warnings

    @property
    def is_stale(self) -> bool:
        return bool(self.stale)

    @property
    def is_narrative(self) -> bool:
        return bool(self.narrative)


def audit_claim(claim: Claim, text: str) -> ClaimResult:
    res = ClaimResult(claim)
    matches = list(re.finditer(claim.pattern, text, claim.flags))
    if len(matches) == 0:
        res.error = "pattern not found (prose may have been reworded -- update the regex)"
        return res
    if claim.occurrence is not None:
        if claim.occurrence >= len(matches):
            res.error = (f"occurrence {claim.occurrence} requested but pattern "
                         f"matched only {len(matches)} time(s)")
            return res
        m = matches[claim.occurrence]
    elif len(matches) > 1:
        res.error = f"pattern matched {len(matches)} times (anchor not unique)"
        return res
    else:
        m = matches[0]
    res.match = m
    res.paper = {g: m.group(g) for g in claim.fmt}

    try:
        res.cache_f = claim.compute()
    except Exception as e:  # noqa: BLE001
        res.error = f"compute() failed: {e}"
        return res
    res.cache_s = {g: (claim.fmt[g](res.cache_f[g]) if callable(claim.fmt[g])
                       else claim.fmt[g].format(res.cache_f[g])) for g in claim.fmt}

    paper_f = {}
    for g in claim.fmt:
        if res.paper[g] != res.cache_s[g]:
            res.stale[g] = (res.paper[g], res.cache_s[g])
        try:
            paper_f[g] = float(res.paper[g])
        except ValueError:
            paper_f[g] = float("nan")
        # soft note for a large relative move even if the format still matches
        denom = max(abs(paper_f[g]), 1e-9)
        if abs(res.cache_f[g] - paper_f[g]) / denom > claim.rel_note:
            res.notes.append(
                f"{g}: {paper_f[g]:g} -> {res.cache_f[g]:g} "
                f"({100*(res.cache_f[g]-paper_f[g])/denom:+.0f}%)"
            )

    if claim.narrative_guard is not None:
        res.narrative = claim.narrative_guard(paper_f, res.cache_f)
    return res


def rebuild_span(text: str, res: ClaimResult) -> str:
    """Return ``text`` with the matched claim's numeric tokens replaced."""
    m = res.match
    s, e = m.span()
    chunk = m.group(0)
    repls = []
    for g, (_old, new) in res.stale.items():
        gs, ge = m.span(g)
        repls.append((gs - s, ge - s, new))
    for gs, ge, new in sorted(repls, reverse=True):
        chunk = chunk[:gs] + new + chunk[ge:]
    return text[:s] + chunk + text[e:]


# --------------------------------------------------------------------------
# Extra audit: the abstract/intro/conclusion improvement-factor claims.
# Reported only (never auto-edited); each is a bracket spanning prose.
# --------------------------------------------------------------------------
def report_factors() -> None:
    """The improvement factors the abstract and contributions state, checked
    against the caches. A floor claim ("more than fivefold") passes when
    every baseline's factor exceeds it; an "about X" claim passes when
    every baseline's factor is within 0.1 of X."""
    import re as _re
    tex = open(MAIN).read()
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def _floor(pattern):
        m = _re.search(pattern, tex)
        return words.get(m.group(1)) if m else None

    noise_floor = _floor(r"noise KL divergence more than (\w+)fold")
    post_floor = _floor(r"posterior KL divergence more than (\w+)fold")
    further = _re.search(r"factors of about \$([0-9.]+)\$ and \$([0-9.]+)\$ on two further", tex)
    ex2_about, ex3_about = ((float(further.group(1)), float(further.group(2)))
                            if further else (None, None))

    def _line(label, d, floor=None, about=None):
        facs = {k: d[k] / d["cas"] for k in ("pca", "pom", "gauss")}
        lo, hi = min(facs.values()), max(facs.values())
        if floor is not None:
            verdict = (f"every baseline exceeds {floor}" if lo > floor
                       else f"CLAIM 'more than {floor}-fold' FAILS: min {lo:.2f}")
        elif about is not None:
            ok = all(abs(v - about) <= 0.1 for v in facs.values())
            verdict = (f"every baseline within 0.1 of {about}" if ok
                       else f"CLAIM 'about {about}' FAILS: range {lo:.2f}-{hi:.2f}")
        else:
            verdict = "no claim found in the text"
        detail = ", ".join(f"{k.upper()} {v:.2f}x" for k, v in facs.items())
        print(f"    {label:24s}: {detail}   [CAS {d['cas']:.3f} nats; {verdict}]")

    print("\n  Improvement factors over baselines (headline N, vs CAS):")
    _line("noise KL (Example 1)", _section52_kls(), floor=noise_floor)
    _line("posterior KL (Example 1)", _section53_kls(), floor=post_floor)
    for _b, _lab, _c in (("even_fold", "noise KL (Example 2)", ex2_about),
                         ("conformal_cube", "noise KL (Example 3)", ex3_about)):
        try:
            _line(_lab, _section52_kls(_b), about=_c)
        except Exception as _e:  # noqa: BLE001
            print(f"    {_lab:24s}: unavailable ({_e})")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="apply mechanical (non-narrative) updates in place")
    ap.add_argument("--apply-narrative", nargs="*", default=[], metavar="KEY",
                    help="also apply these narrative-flagged claims (after you've "
                         "reviewed and fixed the surrounding prose)")
    ap.add_argument("--only", nargs="*", default=None, metavar="KEY",
                    help="restrict to these claim keys")
    ap.add_argument("--figures", action="store_true",
                    help="regenerate figures from caches (figures/make_all.py) too")
    args = ap.parse_args()

    claims = REGISTRY
    if args.only:
        claims = [c for c in REGISTRY if c.key in set(args.only)]
        if not claims:
            print(f"no claims match {args.only}; known: {[c.key for c in REGISTRY]}")
            return 2

    # group claims by file so we read/write each file once
    by_file: dict[str, list[Claim]] = {}
    for c in claims:
        by_file.setdefault(c.file, []).append(c)

    any_stale = any_narr = any_err = False
    apply_narr = set(args.apply_narrative)

    for fpath, fclaims in by_file.items():
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        rel = os.path.relpath(fpath, REPO)
        print(f"\n{BOLD}=== {rel} ==={RESET}")

        results = [audit_claim(c, text) for c in fclaims]
        new_text = text
        applied = []

        for res in results:
            c = res.claim
            if res.error:
                any_err = True
                print(f"  {RED}[ERROR]{RESET} {c.key:14s} {c.section}\n"
                      f"          {res.error}")
                continue

            if not res.is_stale and not res.is_narrative:
                print(f"  {GREEN}[OK]{RESET}    {c.key:14s} {c.section}")
            else:
                tag = (f"{RED}[STALE]{RESET}" if res.is_stale else f"{GREEN}[OK]{RESET}   ")
                print(f"  {tag} {c.key:14s} {c.section}")
                for g, (old, new) in res.stale.items():
                    print(f"            {g}: {old}  ->  {new}")
                for note in res.notes:
                    print(f"            {DIM}Δ {note}{RESET}")

            if res.is_stale:
                any_stale = True
            if res.is_narrative:
                any_narr = True
                print(f"  {YELLOW}        ⚠ NARRATIVE — review the prose, not just the number:{RESET}")
                for msg in res.narrative:
                    print(f"            - {msg}")

            # decide whether to write this claim
            if args.apply and res.is_stale:
                if res.is_narrative and c.key not in apply_narr:
                    print(f"  {YELLOW}        ↳ held back (narrative). Re-run with "
                          f"--apply-narrative {c.key} once prose is fixed.{RESET}")
                else:
                    # recompute the match against the evolving new_text
                    res2 = audit_claim(c, new_text)
                    if res2.match and res2.is_stale:
                        new_text = rebuild_span(new_text, res2)
                        applied.append(c.key)

        if args.apply and applied:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"  {GREEN}wrote {rel}: applied {applied}{RESET}")

    # factor bracket report (informational)
    if not args.only or "sec52" in set(args.only or []) or "sec53" in set(args.only or []):
        try:
            report_factors()
        except Exception as e:  # noqa: BLE001
            print(f"  (factor report skipped: {e})")

    if args.figures:
        mk = os.path.join(REPO, "figures", "make_all.py")
        print(f"\n{BOLD}=== regenerating figures ==={RESET}")
        if os.path.exists(mk):
            r = subprocess.run([sys.executable, mk], cwd=REPO)
            print(f"  figures/make_all.py exit={r.returncode}")
        else:
            print(f"  {YELLOW}figures/make_all.py not found; skipping{RESET}")

    # summary / exit code
    print(f"\n{BOLD}summary:{RESET} "
          f"{'stale present' if any_stale else 'no stale'}, "
          f"{'narrative flags present' if any_narr else 'no narrative flags'}, "
          f"{'errors present' if any_err else 'no errors'}")
    if args.apply:
        print("  (re-run without --apply to confirm a clean audit, then rebuild the PDFs)")
    return 1 if (any_stale or any_narr or any_err) else 0


if __name__ == "__main__":
    raise SystemExit(main())
