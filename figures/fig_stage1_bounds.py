#!/usr/bin/env python3
"""Part II: Stage-1 a priori terms and a posteriori estimate.
Reads cache/aposteriori_{b}.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (os.path.join(REPO, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from cas.paperstyle import apply_paper_style, top_legend, panel_tag
except Exception:
    def apply_paper_style():
        pass

    def top_legend(ax, ncol, handles=None, labels=None, fontsize=6.4, pad=1.01):
        kw = dict(loc="lower center", bbox_to_anchor=(0.5, pad), ncol=ncol,
                  frameon=False, fontsize=fontsize)
        ax.legend(**kw) if handles is None else ax.legend(handles, labels, **kw)

    def panel_tag(ax, text, fontsize=8):
        ax.text(0.025, 0.97, text, transform=ax.transAxes, ha="left",
                va="top", fontsize=fontsize, fontweight="bold")

C_HERM, C_RANK, C_EST = "#eda100", "#2a78d6", "#c23b3b"
C_TRUE, C_HATE, C_RHS = "#0b0b0b", "#2a78d6", "#1baf7a"
LABELS = {"banana": "Example 1", "even_fold": "Example 2",
          "conformal_cube": "Example 3"}


def load(tag):
    path = os.path.join(REPO, "cache", f"aposteriori_{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def load_dsub(tag):
    """Reference E_proj per N, keyed by N and averaged over seeds, from the
    rank-Gaussianized quadrature cache (delta-sub-cells), so the reference
    line and the stack it is compared against stand at the same N."""
    path = os.path.join(REPO, "cache", f"delta_sub_Z_cells_{tag}.json")
    if not os.path.exists(path):
        return {}
    cells = json.load(open(path))["cells"]
    return {N: float(np.mean([c["delta_sub_Z_mean"]
                              for c in cells if c["N"] == N]))
            for N in sorted({c["N"] for c in cells})}


def panel_a(ax, data, dsub, bar_ns):
    x, ticks = 0.0, []
    for tag in ("banana", "even_fold", "conformal_cube"):
        d = data.get(tag)
        if d is None:
            continue
        herm = np.sqrt(d["hermite_sq"])
        rank = np.sqrt(d["E_r"])
        for N in bar_ns:
            row = next((r for r in d["rows"] if r["N"] == N), None)
            if row is None:
                continue
            est = np.sqrt(max(np.mean(row["penalty"]), 0.0))
            ax.bar(x, herm, 0.8, color=C_HERM)
            ax.bar(x, rank, 0.8, bottom=herm, color=C_RANK)
            ax.bar(x, est, 0.8, bottom=herm + rank, color=C_EST)
            ds = dsub.get(tag, {}).get(N)
            if ds is not None:
                ref = np.sqrt(2.0 * ds)
                ax.plot([x - 0.4, x + 0.4], [ref, ref], color=C_TRUE, lw=1.4,
                        zorder=5)
            ticks.append((x, f"${N:,}$".replace(",", "\\,")))
            x += 1.0
        # One example name per group, centred under its pair of N ticks, so
        # the name is not repeated once per bar.
        if ticks:
            group = [t for t, _ in ticks[-len(bar_ns):]]
            ax.text(sum(group) / len(group), -0.14, LABELS[tag],
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8)
        x += 0.6
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([s for _, s in ticks], fontsize=7)
    ax.set_xlabel(r"sample size $N$", labelpad=18)
    ax.set_ylabel(r"$\sqrt{2\,\mathcal{D}_{\mathrm{KL}}}$ bound terms")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (C_HERM, C_RANK, C_EST)]
    handles.append(plt.Line2D([0], [0], color=C_TRUE, lw=1.4))
    top_legend(ax, 2, handles,
               ["Hermite truncation", "rank truncation",
                "statistical estimation",
                r"reference $\sqrt{2\,\mathcal{E}_{\mathrm{proj}}}$"])
    panel_tag(ax, "(a)")


def panel_b(ax, d, tag):
    Ns = np.array([r["N"] for r in d["rows"]], float)
    res = np.array([r["resB"] for r in d["rows"]], float)
    hatE = np.array([r["hatEB"] for r in d["rows"]], float)
    rhs = np.array([r["rhs"] for r in d["rows"]], float)
    for arr, c, lab, ls in (
            (rhs, C_RHS, r"a posteriori bound ($\delta = 0.05$)", "-"),
            (res, C_TRUE,
             r"quantity bounded: $\mathrm{tr}((I-\widehat{P}_r)C)$", "-"),
            (hatE, C_HATE, r"spectral part $\widehat{E}_r$ alone", "--")):
        ax.plot(Ns, arr.mean(1), ls, color=c, marker="o", ms=3, label=lab)
        ax.fill_between(Ns, arr.min(1), arr.max(1), color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"dimension reduction error (trace units)")
    top_legend(ax, 1)
    panel_tag(ax, f"(b) {LABELS[tag]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="banana", choices=sorted(LABELS))
    ap.add_argument("--bar-ns", default="2500,50000")
    ap.add_argument("--out", default=os.path.join(HERE, "fig_stage1_bounds"))
    args = ap.parse_args()
    apply_paper_style()
    data = {t: load(t) for t in LABELS}
    dsub = {t: load_dsub(t) for t in LABELS}
    if data.get(args.benchmark) is None:
        sys.exit(f"missing cache for {args.benchmark}")
    missing = [t for t in LABELS if not dsub.get(t)]
    if missing:
        print(f"[warn] no E_proj reference for {missing}; the reference line "
              f"is omitted there. Run 'python reproduce.py --recompute part2-decomposition'.")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panel_a(axa, data, dsub, [int(x) for x in args.bar_ns.split(",")])
    panel_b(axb, data[args.benchmark], args.benchmark)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=220, bbox_inches="tight")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
