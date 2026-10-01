#!/usr/bin/env python3
"""Part II: assembled a posteriori estimate against the reference total KL.
Reads cache/apost_e2e_{b}.json and cache/aposteriori_{b}.json.
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
    from cas.paperstyle import apply_paper_style, panel_tag
except Exception:
    def apply_paper_style():
        pass

    def panel_tag(ax, text, fontsize=8):
        ax.text(0.025, 0.97, text, transform=ax.transAxes, ha="left",
                va="top", fontsize=fontsize, fontweight="bold")

C_MARG, C_SUB, C_DEN = "#eda100", "#2a78d6", "#1baf7a"
C_REF, C_REAL = "#0b0b0b", "#c23b3b"
LABELS = {"banana": "Example 1", "even_fold": "Example 2",
          "conformal_cube": "Example 3"}


def series(tag):
    e2e = json.load(open(os.path.join(REPO, "cache", "apost_e2e_%s.json" % tag)))
    ap = json.load(open(os.path.join(REPO, "cache", "aposteriori_%s.json" % tag)))
    bnd = {r["N"]: (0.5 * float(np.mean(r["rhs"])),
                    0.5 * float(np.mean(r["resB"]))) for r in ap["rows"]}
    rows = []
    for N in sorted({r["N"] for r in e2e} & set(bnd)):
        cs = [r for r in e2e if r["N"] == N]
        rows.append(dict(
            N=N,
            marg=float(np.mean([r["dhat_marg"] + r["log_mass"] for r in cs])),
            den=float(np.mean([r["dhat_den"] for r in cs])),
            sub=bnd[N][0],
            realized=bnd[N][1],
            ref=float(np.mean([r["ref_full_kl"] for r in cs])),
            ref_se=float(np.mean([r["ref_full_kl_se"] for r in cs])),
        ))
    return rows


def panel(ax, rows, tag, legend=False):
    x = np.arange(len(rows), dtype=float)
    marg = np.array([r["marg"] for r in rows])
    sub = np.array([r["sub"] for r in rows])
    den = np.array([r["den"] for r in rows])
    ref = np.array([r["ref"] for r in rows])
    rse = np.array([r["ref_se"] for r in rows])
    real = np.array([r["realized"] for r in rows])

    tot = marg + sub + den
    ax.bar(x, marg, 0.62, color=C_MARG, label=r"$\widehat{\mathcal{E}}_{\mathrm{marg}}$")
    ax.bar(x, sub, 0.62, bottom=marg, color=C_SUB,
           label=r"bounded subspace term, $\delta=0.05$")
    ax.bar(x, den, 0.62, bottom=marg + sub, color=C_DEN,
           label=r"$\widehat{\mathcal{E}}_{\mathrm{dens}}$")
    ax.errorbar(x, ref, yerr=rse, fmt="_", color=C_REF, ms=15, mew=1.8,
                capsize=3, lw=1.2, zorder=5, label="reference total")
    ax.plot(x, marg + real + den, ls="none", marker="d", ms=4.5, mfc="none",
            color=C_REAL, zorder=4,
            label=r"estimate at the realized $\frac{1}{2}\mathrm{Tr}$")
    for xi, t, rf in zip(x, tot, ref):
        ax.annotate(r"$%.0f\times$" % (t / rf) if t / rf >= 10
                    else r"$%.1f\times$" % (t / rf),
                    (xi, t), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(["{:,}".format(r["N"]).replace(",", "\\,")
                        for r in rows], fontsize=7)
    ax.set_xlabel(r"sample size $N$")
    panel_tag(ax, LABELS[tag])
    ax.set_yscale("log")
    ax.set_ylim(0.02, 3000)
    if legend:
        ax.set_ylabel("nats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig_apost_endtoend"))
    args = ap.parse_args()
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for i, tag in enumerate(("banana", "even_fold", "conformal_cube")):
        rows = series(tag)
        panel(axes[i], rows, tag, legend=(i == 0))
        eff = [(r["marg"] + r["sub"] + r["den"]) / r["ref"] for r in rows]
        print("%-15s effectivity: %s" % (
            tag, ", ".join("N=%d: %.2f" % (r["N"], e)
                           for r, e in zip(rows, eff))))
        print("%-15s   est=%s" % (tag, ", ".join(
            "%.3f" % (r["marg"] + r["sub"] + r["den"]) for r in rows)))
        print("%-15s   ref=%s" % (tag, ", ".join("%.3f" % r["ref"] for r in rows)))
    fig.tight_layout()
    # One legend for all three panels, above the row and out of the bars,
    # rather than five entries crammed into the first panel at 5.8pt.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=6.4, handlelength=1.5,
               columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=220, bbox_inches="tight")
        print("wrote %s.%s" % (args.out, ext))


if __name__ == "__main__":
    main()
