"""Part I, fig:bip-posteriors: posterior grids of the true noise model, CAS,
PCA-HCSM, the product of marginals and the Gaussian copula for one observation
at N = 50,000. Reads cache/run2026/{example}/hero_N50000.npz; writes
bip_posteriors.png.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC


HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas.paperstyle import (  # noqa: E402
    COLOR_CAS, COLOR_PCA, COLOR_GAUSS, COLOR_POM, COLOR_TRUE,
    apply_paper_style,
)


_CACHE_DIR = os.environ.get(
    "V82_CACHE_DIR", os.path.join(_REPO, "cache")
)
BENCHMARKS = ["banana", "even_fold", "conformal_cube"]
DEFAULT_HERO = "25"   # the draw whose true posterior places visible mass on
                      # both preimage branches (sec:exp-bip); CAS reproduces
                      # both modes, the baselines neither.


def _cache_path(benchmark: str) -> str:
    return os.path.join(_CACHE_DIR, "run2026", benchmark, "hero_N50000.npz")


def _default_out(benchmark: str) -> str:
    name = ("bip_posteriors.png" if benchmark == "banana"
            else f"bip_posteriors_{benchmark}.png")
    return os.environ.get("BIP_POSTERIORS_OUT", os.path.join(HERE, name))


def make_cmap(color: str, name: str, start_alpha: float = 0.10,
              dark_factor: float = 0.65) -> LSC:
    """Light tint at low end -> darker than base color at high end."""
    rgb = np.array(mc.to_rgb(color))
    light = (1 - start_alpha) * np.ones(3) + start_alpha * rgb
    dark = dark_factor * rgb
    return LSC.from_list(name, [(0.0, light), (1.0, dark)])


def _auto_zoom(ps, X1, X2, frac: float = 0.02, pad: float = 0.6):
    """(xlim, ylim) bounding box of the region where any panel exceeds ``frac``
    of its max, padded -- so the framing adapts to the benchmark and the truth
    x* instead of a hard-coded window."""
    mask = np.zeros_like(ps[0], dtype=bool)
    for p in ps:
        mask |= (p >= frac * p.max())
    xs, ys = X1[mask], X2[mask]
    return ((float(xs.min()) - pad, float(xs.max()) + pad),
            (float(ys.min()) - pad, float(ys.max()) + pad))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 5 / SM BIP hero posteriors")
    ap.add_argument("--benchmark", default="banana", choices=BENCHMARKS,
                    help="noise law (banana = main-text Fig 5)")
    ap.add_argument("--hero", default=DEFAULT_HERO,
                    help="hero trial index to render")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()
    cache_path = _cache_path(args.benchmark)
    out = args.out or _default_out(args.benchmark)

    apply_paper_style()
    plt.rcParams.update({"axes.titlesize": 11})

    z = np.load(cache_path, allow_pickle=True)
    trials = z["trials"].item()
    r = trials[args.hero]
    kl_cas, kl_pca, kl_gauss, kl_pom = (
        r["kl_cas"], r["kl_pca"], r["kl_gauss"], r["kl_pom"]
    )
    print(f"hero {args.hero}: cas={kl_cas:.2f} pca={kl_pca:.2f} "
          f"gauss={kl_gauss:.2f} pom={kl_pom:.2f}")

    grid = np.linspace(-6, 6, 80)
    X1, X2 = np.meshgrid(grid, grid)
    x_star = r["x_star"]

    p_true  = r["p_true"].reshape(80, 80)
    p_cas   = r["p_cas"].reshape(80, 80)
    p_pca   = r["p_pca"].reshape(80, 80)
    p_gauss = r["p_gauss"].reshape(80, 80)
    p_pom   = r["p_pom"].reshape(80, 80)

    # Panels in display order: True posterior anchors top-left as the reference.
    panels = [
        ("True posterior", None,     p_true,  COLOR_TRUE),
        ("CAS",            kl_cas,   p_cas,   COLOR_CAS),
        ("PCA-HCSM",       kl_pca,   p_pca,   COLOR_PCA),
        ("PoM",            kl_pom,   p_pom,   COLOR_POM),
        ("Gaussian copula", kl_gauss, p_gauss, COLOR_GAUSS),
    ]

    fig, axes_flat = plt.subplots(2, 3, figsize=(7.0, 4.0))
    axes = axes_flat.flatten()

    # Auto-zoom to the posterior mass so the framing adapts to the benchmark
    # and the truth x* (the old hard-coded window was tuned to a different x*).
    XLIM, YLIM = _auto_zoom([p_true, p_cas, p_pca, p_pom, p_gauss], X1, X2)

    for ax, (label, kl, p, color) in zip(axes[:5], panels):
        cmap = make_cmap(
            color, label.replace(" ", "_"),
            start_alpha=0.06 if label == "True posterior" else 0.10,
            dark_factor=0.85 if label == "True posterior" else 0.65,
        )
        vmax = p.max()
        levels = np.linspace(0.05 * vmax, vmax, 18)
        ax.contourf(X1, X2, p, levels=levels, cmap=cmap, extend="neither")
        ax.plot(x_star[0], x_star[1], "*", color="red", markersize=10,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.grid(False)

        # Title above the panel — method name in method colour, KL trailing
        if kl is None:
            title = label
        else:
            title = (rf"{label}\quad $D_{{\mathrm{{KL}}}} = {kl:.2f}$"
                     if plt.rcParams.get("text.usetex", False)
                     else f"{label}    $D_{{KL}} = {kl:.2f}$")
        ax.set_title(title, color=color, pad=4, fontweight="bold")

    # Bottom row: x-axis labels on (1,0) and (1,1); also (0,2) since (1,2) is empty
    axes_flat[0, 2].set_xlabel(r"$x_1$")
    axes_flat[1, 0].set_xlabel(r"$x_1$")
    axes_flat[1, 1].set_xlabel(r"$x_1$")
    # y-axis label on left column only (top + bottom)
    axes_flat[0, 0].set_ylabel(r"$x_2$")
    axes_flat[1, 0].set_ylabel(r"$x_2$")
    # Strip y-tick labels from middle/right columns to reduce clutter
    for r in (0, 1):
        for c in (1, 2):
            axes_flat[r, c].tick_params(axis="y", labelleft=False)
    # Strip x-tick labels from top-row middle (PCA-HCSM is the rightmost top panel,
    # which keeps its labels because nothing is below it).
    axes_flat[0, 0].tick_params(axis="x", labelbottom=False)
    axes_flat[0, 1].tick_params(axis="x", labelbottom=False)

    # Empty cell: short legend explaining the reference and the marker.
    leg_ax = axes[5]
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    leg_ax.set_xlabel("")
    leg_ax.set_ylabel("")
    for spine in leg_ax.spines.values():
        spine.set_visible(False)

    # Star marker + label
    leg_ax.plot([0.10], [0.65], marker="*", color="red", markersize=12,
                markeredgecolor="black", markeredgewidth=0.6,
                transform=leg_ax.transAxes)
    leg_ax.text(0.20, 0.65, r"$x^\star$ (true parameter)",
                transform=leg_ax.transAxes, va="center", ha="left", fontsize=10)

    # Lower D_KL = closer to truth
    leg_ax.text(
        0.10, 0.40,
        r"Lower $D_{\mathrm{KL}} \Rightarrow$ closer to true posterior",
        transform=leg_ax.transAxes, va="center", ha="left",
        fontsize=10, color="0.25",
    )

    plt.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
