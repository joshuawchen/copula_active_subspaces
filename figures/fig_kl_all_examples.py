"""Part I, fig:testll and fig:bip-kl: noise and posterior KL divergence versus N
for Examples 1-3. --which testll writes test_ll_all.png and --which bip writes
bip_kl_all.png, from cache/run2026/{example}/{testll,posterior}_N{N}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas.paperstyle import (  # noqa: E402
    COLOR_CAS, COLOR_PCA, COLOR_GAUSS, COLOR_POM,
    apply_paper_style,
)


CACHE_DIR = os.environ.get("V82_CACHE_DIR", os.path.join(_REPO, "cache"))
YSCALE = os.environ.get("KL_YSCALE", "log")

BENCHES = ["banana", "even_fold", "conformal_cube"]
PANEL_TITLES = {"banana": "Example 1", "even_fold": "Example 2",
                "conformal_cube": "Example 3"}
N_LIST = [500, 1000, 2000, 5000, 10000, 20000, 50000]
N_LABELS = {500: "500", 1000: "1k", 2000: "2k", 5000: "5k",
            10000: "10k", 20000: "20k", 50000: "50k"}
METHODS = ["PoM", "Gaussian", "PCA", "CAS"]
LABELS = {"PoM": "PoM", "Gaussian": "Gaussian copula",
          "PCA": "PCA-HCSM", "CAS": "CAS"}
COLORS = {"PoM": COLOR_POM, "Gaussian": COLOR_GAUSS,
          "PCA": COLOR_PCA, "CAS": COLOR_CAS}

KINDS = {
    "testll": dict(
        fname="testll_N{N}.json",
        out="test_ll_all.png",
        ylabel=(r"$D_{\mathrm{KL}}(\pi_{\mathbf{n}} \,\|\, \hat\pi_{\mathbf{n}})"
                r"\ \downarrow$" + "\n[nats]"),
    ),
    "bip": dict(
        fname="posterior_N{N}.json",
        out="bip_kl_all.png",
        ylabel=(r"$D_{\mathrm{KL}}(\pi(\cdot\,|\,y) \,\|\,"
                r" \hat\pi(\cdot\,|\,y))\ \downarrow$" + "\n[nats]"),
    ),
}


def cache_path(kind: str, benchmark: str, N: int) -> str:
    return os.path.join(CACHE_DIR, "run2026", benchmark,
                        KINDS[kind]["fname"].format(N=N))


def load_cell(kind: str, benchmark: str, N: int) -> dict[str, np.ndarray]:
    with open(cache_path(kind, benchmark, N)) as f:
        d = json.load(f)
    return {m: np.array([d[m][s] for s in sorted(d[m], key=int)])
            for m in METHODS if d.get(m)}


def draw_panel(ax, data_by_N: dict, jitter_seed: int) -> None:
    """One benchmark's box/scatter panel, layout as in the single-panel
    figures (CAS drawn last so its boxes sit on top at small N)."""
    n_methods = len(METHODS)
    n_N = len(N_LIST)
    group_width = 0.78
    box_width = group_width / n_methods
    positions = {
        m: [j + (i - 1.5) * box_width for j in range(n_N)]
        for i, m in enumerate(METHODS)
    }
    jitter = np.random.default_rng(jitter_seed)
    for m in ["PoM", "Gaussian", "PCA", "CAS"]:
        box_data = [data_by_N[N][m] for N in N_LIST]
        z_box = 5 if m == "CAS" else 3
        z_pts = z_box + 1
        ax.boxplot(
            box_data,
            positions=positions[m],
            widths=box_width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=COLORS[m], edgecolor="black",
                          linewidth=0.5),
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=0.5),
            capprops=dict(color="black", linewidth=0.5),
            flierprops=dict(marker="", markersize=0),
            showmeans=False,
            zorder=z_box,
        )
        means = [float(np.mean(v)) for v in box_data]
        ax.scatter(positions[m], means, marker="D", s=12,
                   facecolor="white", edgecolor="black", linewidth=0.6,
                   zorder=z_pts + 0.5)
        for j, vals in enumerate(box_data):
            x = positions[m][j] + jitter.uniform(-0.015, 0.015, len(vals))
            ax.scatter(x, vals, color=COLORS[m], alpha=0.40, s=5,
                       linewidths=0, zorder=z_pts)
    ax.set_xticks(range(n_N))
    ax.set_xticklabels([N_LABELS[N] for N in N_LIST])
    ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", length=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="3-panel all-examples KL figure (test-LL or BIP)")
    ap.add_argument("--which", required=True, choices=sorted(KINDS))
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()
    kind = args.which
    out = args.out or os.path.join(HERE, KINDS[kind]["out"])

    missing = [cache_path(kind, b, N) for b in BENCHES for N in N_LIST
               if not os.path.exists(cache_path(kind, b, N))]
    if missing:
        print(f"[skip] {kind}: {len(missing)} cache file(s) missing, "
              f"first: {missing[0]}")
        return

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)

    tables = {}
    for k, (ax, b) in enumerate(zip(axes, BENCHES)):
        data_by_N = {N: load_cell(kind, b, N) for N in N_LIST}
        tables[b] = data_by_N
        draw_panel(ax, data_by_N, jitter_seed=k)
        ax.set_title(PANEL_TITLES[b])
        ax.set_xlabel(r"Sample size $N$")
    axes[0].set_ylabel(KINDS[kind]["ylabel"])
    if YSCALE == "log":
        axes[0].set_yscale("log")

    patches = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.5,
                     label=LABELS[m]) for m in METHODS]
    # House convention for row figures: one figure-level legend above the
    # panels (matches fig03_recovery_row / figS_recovery_ex23), never inside
    # a data region.
    fig.legend(
        handles=patches, loc="lower center", ncol=4, frameon=False,
        handlelength=1.0, handleheight=1.0, columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.99),
    )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out}")
    for b in BENCHES:
        print(f"\n{PANEL_TITLES[b]} ({b})")
        print(f"{'N':>6} {'PoM':>8} {'Gauss':>8} {'PCA':>8} {'CAS':>8}")
        for N in N_LIST:
            d = tables[b][N]
            print(f"{N:>6} {d['PoM'].mean():>8.3f} "
                  f"{d['Gaussian'].mean():>8.3f} {d['PCA'].mean():>8.3f} "
                  f"{d['CAS'].mean():>8.3f}  (seeds: {len(d['CAS'])})")


if __name__ == "__main__":
    main()
