#!/usr/bin/env python3
"""Part II: composite KL decomposition for Example 1, as a mean over runs and
per run. Reads the per-seed output of experiments/exp_stage2_oracle_kl.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from cas.paperstyle import apply_paper_style
except Exception:
    def apply_paper_style():
        pass

C_SUB = "#2a78d6"
C_MARG = "#eda100"
C_DEN = "#1baf7a"
C_TOT = "#0b0b0b"


def _load(benchmark, out_dir):
    path = os.path.join(out_dir, f"{benchmark}_production",
                        "exp_stage2_oracle_kl.json")
    with open(path) as f:
        return json.load(f), path


def _means(rows):
    Ns = [r["N"] for r in rows]
    sub = np.array([r.get("delta_sub_mean", np.nan) for r in rows])
    marg = np.array([r["delta_marg_mean"] for r in rows])
    den = np.array([r["delta_den_mean"] for r in rows])
    full = np.array([r["full_kl_mean"] for r in rows])
    return Ns, sub, marg, den, full


def _order_md(marg, den):
    marg = np.clip(np.asarray(marg, float), 0, None)
    den = np.clip(np.asarray(den, float), 0, None)
    marg_big = marg >= den
    mid_h = np.where(marg_big, marg, den)
    top_h = np.where(marg_big, den, marg)
    mid_c = [C_MARG if b else C_DEN for b in np.atleast_1d(marg_big)]
    top_c = [C_DEN if b else C_MARG for b in np.atleast_1d(marg_big)]
    return mid_h, mid_c, top_h, top_c


def _panel_mean(ax, Ns, sub, marg, den, full):
    x = np.arange(len(Ns))
    s = np.clip(sub, 0, None)
    mid_h, mid_c, top_h, top_c = _order_md(marg, den)
    w = 0.62
    ax.bar(x, s, w, color=C_SUB)
    ax.bar(x, mid_h, w, bottom=s, color=mid_c)
    ax.bar(x, top_h, w, bottom=s + mid_h, color=top_c)
    ax.plot(x, full, marker="o", ms=6, color=C_TOT, lw=1.3, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in Ns], rotation=30, ha="right")
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel("KL (nats)")
    ax.set_title("(a)", fontsize=8, loc="left")
    ax.margins(x=0.04)


def _panel_perseed(ax, rows):
    Ns = [r["N"] for r in rows]
    for gi, r in enumerate(rows):
        marg = np.clip(np.array(r["delta_marg_per_seed"], float), 0, None)
        den = np.clip(np.array(r["delta_den_per_seed"], float), 0, None)
        full = np.array(r["full_kl_per_seed"], float)
        sub_raw = r.get("delta_sub_per_seed", [])
        have_sub = len(sub_raw) > 0
        sub = np.clip(np.array(sub_raw, float), 0, None) if have_sub \
            else np.zeros(len(marg))
        n = min(len(sub), len(marg), len(den), len(full))
        sub, marg, den, full = sub[:n], marg[:n], den[:n], full[:n]
        if marg.mean() >= den.mean():
            mid_h, mid_c, top_h, top_c = marg, C_MARG, den, C_DEN
        else:
            mid_h, mid_c, top_h, top_c = den, C_DEN, marg, C_MARG
        span = 0.74
        offs = np.linspace(-span / 2, span / 2, n) if n > 1 else np.array([0.0])
        bw = min(0.085, (span / max(n - 1, 1)) * 0.62)
        xs = gi + offs
        ax.bar(xs, sub, bw, color=C_SUB)
        ax.bar(xs, mid_h, bw, bottom=sub, color=mid_c)
        ax.bar(xs, top_h, bw, bottom=sub + mid_h, color=top_c)
        for xi, f in zip(xs, full):
            ax.plot([xi - bw / 2 - 0.009, xi + bw / 2 + 0.009], [f, f],
                    color=C_TOT, lw=2.3, solid_capstyle="butt", zorder=6)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{int(n):,}" for n in Ns], rotation=30, ha="right")
    ax.set_xlabel(r"sample size $N$")
    ax.set_title("(b)", fontsize=8, loc="left")
    ax.margins(x=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="banana")
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "out"))
    ap.add_argument("--save_dir", default=os.path.join(_REPO, "paper"))
    args = ap.parse_args()

    apply_paper_style()
    summary, path = _load(args.benchmark, args.out_dir)
    rows = summary["per_N"]
    Ns, sub, marg, den, full = _means(rows)
    n_full = rows[0].get("n_cells", "?")
    n_sub = rows[0].get("delta_sub_n_cells", "?")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), sharey=True)
    _panel_mean(axes[0], Ns, sub, marg, den, full)
    _panel_perseed(axes[1], rows)

    handles = [
        Patch(color=C_SUB, label="subspace"),
        Patch(color=C_MARG, label="marginals"),
        Patch(color=C_DEN, label="reduced"),
        Line2D([0], [0], color=C_TOT, marker="o", ms=5, lw=1.3,
               label=r"total $D_{\mathrm{KL}}(\pi_n\,\|\,\hat\pi_n)$"),
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=8, handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    base = os.path.join(args.save_dir, f"fig_composite_main_{args.benchmark}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    print(f"wrote {base}.pdf and {base}.png")
    print(f"  source: {path}")
    print(f"  full_kl seeds/cell: {n_full};  delta_sub seeds/cell: {n_sub}")


if __name__ == "__main__":
    main()
