"""Part I supplement, fig:eigvals-all: eigenvalues of C_hat at N = 50,000 for
Examples 1-3, median and range over five fits. Writes eigvals_all.png.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas.paperstyle import apply_paper_style, COLOR_CAS, COLOR_RANDOM  # noqa: E402
from cas.stage1 import cas_fit_streaming  # noqa: E402


# ----------------------------------------------------------------------------
# Benchmark registry: sampler + (K, q) + r⋆ + d for axis limits
# ----------------------------------------------------------------------------
def _benchmark_specs():
    """Lazy import the samplers (each pulls heavy module-load tables)."""
    from cas import sample_noise as sample_banana
    from cas import sample_even_fold_noise, R_EF_TARGET
    from cas import sample_conformal_cube_noise, R_C3_TARGET
    from cas.config import R_ORACLE as R_BANANA

    return {
        "banana": {
            "sample_fn": sample_banana,
            "d": 20,
            "r_star": R_BANANA,  # 4
            "K": 4,
            "q": 2,
            "label": r"Example~1 ($d=20$, $r^\star=4$)",
        },
        "even_fold": {
            "sample_fn": sample_even_fold_noise,
            "d": 20,
            "r_star": R_EF_TARGET,  # 4
            "K": 4,
            "q": 2,
            "label": r"Example~2 ($d=20$, $r^\star=4$)",
        },
        "conformal_cube": {
            "sample_fn": sample_conformal_cube_noise,
            "d": 20,
            "r_star": R_C3_TARGET,  # 4
            "K": 4,
            "q": 2,
            "label": r"Example~3 ($d=20$, $r^\star=4$)",
        },
    }


# ----------------------------------------------------------------------------
# Compute eigenvalues for one benchmark, multiple seeds
# ----------------------------------------------------------------------------
def compute_eigvals(spec: dict, N: int, n_seeds: int) -> np.ndarray:
    """Returns array of shape (n_seeds, d) of eigenvalues, sorted descending
    per seed."""
    sample_fn = spec["sample_fn"]
    d = spec["d"]
    K = spec["K"]
    q = spec["q"]

    eigs_all = np.empty((n_seeds, d), dtype=np.float64)
    for s in range(n_seeds):
        t0 = time.time()
        rng = np.random.default_rng(s)
        eta = sample_fn(N, rng)
        # We want all eigenvalues — pass r=d so cas_fit returns full spectrum
        result = cas_fit_streaming(eta, K=K, q=q, r=d, lambda0=1e-6)
        eigs = np.asarray(result["eigvals_all"], dtype=np.float64)
        # eigvals_all should already be sorted descending; ensure
        eigs = np.sort(eigs)[::-1]
        if eigs.size != d:
            # Some pipelines return only top eigenvalues. Pad with zeros.
            full = np.zeros(d)
            full[:eigs.size] = eigs
            eigs = full
        eigs_all[s] = eigs
        print(f"  seed {s}: {time.time() - t0:.1f}s, λ_1={eigs[0]:.3g}, "
              f"λ_{spec['r_star']}/λ_{spec['r_star']+1}="
              f"{eigs[spec['r_star']-1] / max(eigs[spec['r_star']], 1e-30):.2f}")
    return eigs_all


def _draw_panel(ax, spec: dict, eigs_all: np.ndarray, *, fs_axis=9, fs_tick=8,
                fs_note=7, show_ylabel=True) -> None:
    """Draw one eigenvalue panel into an existing axis.

    Factored out of render_panel so the same drawing code serves both the
    single-panel PNGs and the combined three-panel figure. Font sizes are
    arguments because the two callers are scaled differently on the page.
    """
    d = spec["d"]
    r_star = spec["r_star"]
    indices = np.arange(1, d + 1)

    median = np.median(eigs_all, axis=0)
    lo = np.min(eigs_all, axis=0)
    hi = np.max(eigs_all, axis=0)
    bulk_floor = np.median(eigs_all[:, r_star:], axis=0).mean()
    mass_top = np.median(eigs_all[:, :r_star].sum(axis=1)
                         / eigs_all.sum(axis=1)) * 100.0
    eigengap = np.median(eigs_all[:, r_star - 1]
                         / np.maximum(eigs_all[:, r_star], 1e-30))

    ax.axhline(bulk_floor, color=COLOR_RANDOM, linestyle=":", linewidth=1.0,
               alpha=0.8, zorder=2, label=r"bulk mean ($k>r^\star$)")
    ax.axvline(r_star + 0.5, color="k", linestyle="--", linewidth=1.0,
               alpha=0.55, zorder=2)
    ax.fill_between(indices, np.maximum(lo, 1e-30), hi,
                    color=COLOR_CAS, alpha=0.18, linewidth=0, zorder=3)
    ax.plot(indices[:r_star], median[:r_star], "o-", color=COLOR_CAS,
            markersize=5.5, linewidth=1.2,
            label=r"$k \leq r^\star$ (active)", zorder=5)
    ax.plot(indices[r_star:], median[r_star:], "o-", color=COLOR_CAS,
            markersize=3.5, linewidth=1.0, markerfacecolor="white",
            markeredgecolor=COLOR_CAS, markeredgewidth=0.9,
            label=r"$k > r^\star$ (bulk)", zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(0.3, d + 0.7)
    ymax_lim = hi[0] * 3.0
    ymin_lim = max(min(median.min(), bulk_floor) * 0.3, 1e-6)
    ax.set_ylim(ymin_lim, ymax_lim)
    ax.annotate(rf"$r^\star={r_star}$", xy=(r_star + 0.7, ymax_lim * 0.5),
                ha="left", va="top", fontsize=fs_note, zorder=5, color="0.25")

    stats_text = (rf"top-$r^\star$: ${mass_top:.0f}\%$" + "\n"
                  + rf"$\widehat{{\lambda}}_{{r^\star}}/\widehat{{\lambda}}_{{r^\star+1}}"
                    rf"={eigengap:.2f}$")
    ax.text(0.03, 0.04, stats_text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=fs_note,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.6", linewidth=0.4, alpha=0.92))

    ax.set_xlabel(r"eigenvalue index $k$", fontsize=fs_axis)
    if show_ylabel:
        ax.set_ylabel(r"$\widehat{\lambda}_k(\widehat{\boldsymbol{C}})$",
                      fontsize=fs_axis)
    ax.set_title(spec["label"], fontsize=fs_axis)
    ax.tick_params(axis="both", which="major", labelsize=fs_tick)
    tick_step = 5 if d < 30 else 10
    xticks = [1] + list(range(tick_step, d + 1, tick_step))
    if xticks[-1] != d:
        xticks.append(d)
    ax.set_xticks(xticks)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=fs_note, frameon=True,
              framealpha=0.92, edgecolor="0.5", handlelength=1.5,
              handletextpad=0.5, borderpad=0.3, labelspacing=0.3)


def render_combined(specs: dict, eigs_by_name: dict, out_path: str) -> None:
    """Three panels in one row, sized for inclusion at full textwidth.

    The per-example PNGs are 4.4in wide and the supplement includes them at
    0.32 textwidth (1.64in), a 0.37 downscale that renders their 7-9pt text
    at 2.6-3.4pt. This figure is 7.2in wide for inclusion at 5.125in, a 0.71
    downscale, matching the other three-panel figures of the paper.
    """
    apply_paper_style()
    names = ["banana", "even_fold", "conformal_cube"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for j, (name, ax) in enumerate(zip(names, axes)):
        _draw_panel(ax, specs[name], eigs_by_name[name], show_ylabel=(j == 0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------------
# Render one panel
# ----------------------------------------------------------------------------
def render_panel(bench_name: str, spec: dict, eigs_all: np.ndarray,
                 out_path: str, N: int) -> None:
    apply_paper_style()
    d = spec["d"]
    r_star = spec["r_star"]
    indices = np.arange(1, d + 1)

    median = np.median(eigs_all, axis=0)
    lo = np.min(eigs_all, axis=0)
    hi = np.max(eigs_all, axis=0)

    # Bulk floor: mean of eigenvalues r_star+1 ... d (median across seeds)
    bulk_floor = np.median(eigs_all[:, r_star:], axis=0).mean()
    # Fraction of mass captured by top-r⋆ (median across seeds)
    mass_top = np.median(eigs_all[:, :r_star].sum(axis=1)
                         / eigs_all.sum(axis=1)) * 100.0
    # Eigengap λ_{r*}/λ_{r*+1}, median across seeds
    eigengap = np.median(eigs_all[:, r_star - 1] / np.maximum(eigs_all[:, r_star], 1e-30))

    # Sized for inclusion at 0.6\textwidth (~3.6 inches wide in print).
    # The aspect roughly matches the half-width inline reference figures
    # used elsewhere in the supplement.
    fig, ax = plt.subplots(figsize=(4.4, 2.4))

    # Bulk floor horizontal line first (background)
    ax.axhline(bulk_floor, color=COLOR_RANDOM, linestyle=":", linewidth=1.0,
               alpha=0.8, zorder=2,
               label=rf"bulk mean ($k>r^\star$)")

    # Cutoff vertical line at r⋆
    ax.axvline(r_star + 0.5, color="k", linestyle="--", linewidth=1.0,
               alpha=0.55, zorder=2)

    # Shaded band [min, max] across seeds
    ax.fill_between(indices, np.maximum(lo, 1e-30), hi,
                    color=COLOR_CAS, alpha=0.18, linewidth=0, zorder=3)

    # Distinct markers: top-r⋆ filled, bulk hollow
    ax.plot(indices[:r_star], median[:r_star], "o-", color=COLOR_CAS,
            markersize=5.5, linewidth=1.2,
            label=r"$k \leq r^\star$ (active)", zorder=5)
    ax.plot(indices[r_star:], median[r_star:], "o-", color=COLOR_CAS,
            markersize=3.5, linewidth=1.0,
            markerfacecolor="white", markeredgecolor=COLOR_CAS,
            markeredgewidth=0.9,
            label=r"$k > r^\star$ (bulk)", zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(0.3, d + 0.7)
    ymax_lim = hi[0] * 3.0
    ymin_lim = max(min(median.min(), bulk_floor) * 0.3, 1e-6)
    ax.set_ylim(ymin_lim, ymax_lim)

    # r⋆ cutoff annotation, just to the right of the vertical line at top
    ax.annotate(rf"$r^\star={r_star}$",
                xy=(r_star + 0.7, ymax_lim * 0.5),
                ha="left", va="top", fontsize=8, zorder=5,
                color="0.25")

    # Stats box: place in lower-left so it doesn't collide with the
    # curve or the upper-right legend
    stats_text = (rf"top-$r^\star$: ${mass_top:.0f}\%$" + "\n"
                  + rf"$\widehat{{\lambda}}_{{r^\star}}/\widehat{{\lambda}}_{{r^\star+1}}"
                    rf"={eigengap:.2f}$")
    ax.text(0.03, 0.04, stats_text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.6", linewidth=0.4, alpha=0.92))

    ax.set_xlabel(r"eigenvalue index $k$", fontsize=9)
    ax.set_ylabel(r"$\widehat{\lambda}_k(\widehat{\boldsymbol{C}})$", fontsize=9)
    ax.set_title(spec["label"], fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    # Force integer eigenvalue-index ticks at multiples of 5 (10 for d ≥ 25)
    tick_step = 5 if d < 30 else 10
    xticks = [1] + list(range(tick_step, d + 1, tick_step))
    if xticks[-1] != d:
        xticks.append(d)
    ax.set_xticks(xticks)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True,
              framealpha=0.92, edgecolor="0.5",
              handlelength=1.5, handletextpad=0.5,
              borderpad=0.3, labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["banana", "even_fold", "conformal_cube"],
                        help="one benchmark only; default = all")
    parser.add_argument("--all", action="store_true",
                        help="render all three benchmarks")
    parser.add_argument("--N", type=int, default=50000,
                        help="sample size for Stage-1 fits (default 50000)")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    specs = _benchmark_specs()
    if args.all or args.benchmark is None:
        names = ["banana", "even_fold", "conformal_cube"]
    else:
        names = [args.benchmark]

    os.makedirs(args.out_dir, exist_ok=True)
    eigs_by_name = {}
    for name in names:
        spec = specs[name]
        print(f"=== {name} (d={spec['d']}, r⋆={spec['r_star']}, "
              f"N={args.N}, {args.n_seeds} seeds) ===")
        eigs = compute_eigvals(spec, args.N, args.n_seeds)
        eigs_by_name[name] = eigs
        out_path = os.path.join(args.out_dir, f"eigvals_{name}.png")
        render_panel(name, spec, eigs, out_path, args.N)

    if len(eigs_by_name) == 3:
        render_combined(specs, eigs_by_name,
                        os.path.join(args.out_dir, "eigvals_all.png"))


if __name__ == "__main__":
    main()
