"""Part I supplement, fig:random-subspace: subspace-only KL at the CAS, PCA,
reference and random bases. Reads cache/exp_oracle_kl_random_subspace.npz;
writes random_subspace.png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas.paperstyle import (  # noqa: E402
    COLOR_CAS, COLOR_RANDOM, apply_paper_style,
)

try:
    from cas.paperstyle import COLOR_PCA  # noqa: E402
except Exception:
    COLOR_PCA = "#009E73"


DEFAULT_CACHE_PATHS = [
    "exp_oracle_kl_random_subspace.npz",
    os.path.join(_REPO, "cache", "exp_oracle_kl_random_subspace.npz"),
]

BENCHMARK_TITLES = {
    "banana":       r"Example~1",
    "even_fold":      r"Example~2",
    "conformal_cube": r"Example~3",
}
BENCHMARK_ORDER = ("banana", "even_fold", "conformal_cube")


def _locate_cache() -> str | None:
    for p in DEFAULT_CACHE_PATHS:
        if os.path.exists(p):
            return p
    return None


def _aggregate_cells(cells: dict, benchmark: str):
    """Return dict {N: {'oracle': arr, 'cas': arr, 'pca': arr, 'rand': arr}}."""
    rows = [k for k in cells if k[0] == benchmark]
    N_values = sorted({k[1] for k in rows})
    seeds = sorted({k[2] for k in rows})
    out = {}
    for N in N_values:
        oracle = [cells[(benchmark, N, s, "oracle")] for s in seeds
                  if (benchmark, N, s, "oracle") in cells]
        cas    = [cells[(benchmark, N, s, "cas")] for s in seeds
                  if (benchmark, N, s, "cas") in cells]
        pca    = [cells[(benchmark, N, s, "pca")] for s in seeds
                  if (benchmark, N, s, "pca") in cells]
        rand = []
        for s in seeds:
            j = 0
            while (benchmark, N, s, ("random", j)) in cells:
                rand.append(cells[(benchmark, N, s, ("random", j))])
                j += 1
        out[N] = dict(
            oracle=np.asarray(oracle),
            cas=np.asarray(cas),
            pca=np.asarray(pca),
            rand=np.asarray(rand),
        )
    return out


def _census(cells: dict, benchmarks, expect_seeds: int, expect_random: int):
    """Report the (benchmark, N) cell census and whether the grid is complete.

    The plotting code below aggregates whatever cells exist, so a cache that is
    still being written yields a figure that looks finished but rests on fewer
    draws. This returns (complete, lines) so the caller can refuse to write.
    """
    lines = []
    complete = True
    for bench in benchmarks:
        rows = [k for k in cells if k[0] == bench]
        N_values = sorted({k[1] for k in rows})
        if not N_values:
            lines.append(f"  {bench:<16} NO CELLS")
            complete = False
            continue
        for N in N_values:
            seeds = sorted({k[2] for k in rows if k[1] == N})
            n_or = sum(1 for s in seeds if (bench, N, s, "oracle") in cells)
            n_cas = sum(1 for s in seeds if (bench, N, s, "cas") in cells)
            n_pca = sum(1 for s in seeds if (bench, N, s, "pca") in cells)
            n_rand = 0
            for s in seeds:
                j = 0
                while (bench, N, s, ("random", j)) in cells:
                    n_rand += 1
                    j += 1
            want_rand = expect_seeds * expect_random
            ok = (n_or >= expect_seeds and n_cas >= expect_seeds
                  and n_pca >= expect_seeds and n_rand >= want_rand)
            complete &= ok
            lines.append(
                f"  {bench:<16} N={N:<7} oracle {n_or:>3}/{expect_seeds}"
                f"  cas {n_cas:>3}/{expect_seeds}"
                f"  pca {n_pca:>3}/{expect_seeds}"
                f"  random {n_rand:>5}/{want_rand}"
                f"   {'ok' if ok else 'INCOMPLETE'}"
            )
    return complete, lines


def main(cache_path: str | None = None,
         out_path: str = "random_subspace.png") -> None:
    apply_paper_style()

    cache_path = cache_path or _locate_cache()
    if cache_path is None or not os.path.exists(cache_path):
        print(f"[skip] exp_oracle_kl_random_subspace.npz not found in "
              f"{DEFAULT_CACHE_PATHS}")
        return
    z = np.load(cache_path, allow_pickle=True)
    cells = z["cells"].item()

    benchmarks_present = sorted(
        {k[0] for k in cells},
        key=lambda b: BENCHMARK_ORDER.index(b) if b in BENCHMARK_ORDER else 99,
    )
    if not benchmarks_present:
        print(f"[skip] no benchmark cells in cache")
        return

    expect_seeds = int(os.environ.get("CAS_EXPECT_SEEDS", "10"))
    expect_random = int(os.environ.get("CAS_EXPECT_RANDOM", "100"))
    missing_bench = [b for b in BENCHMARK_ORDER if b not in benchmarks_present]
    complete, lines = _census(cells, benchmarks_present,
                              expect_seeds, expect_random)
    complete = complete and not missing_bench
    print(f"[census] {cache_path}")
    for line in lines:
        print(line)
    for b in missing_bench:
        print(f"  {b:<16} MISSING ENTIRELY")
    print(f"[census] grid {'COMPLETE' if complete else 'INCOMPLETE'} "
          f"(expected {expect_seeds} seeds and {expect_random} random draws "
          f"per cell, on {len(BENCHMARK_ORDER)} example problems)")

    if os.environ.get("CAS_VERIFY_ONLY", "") == "1":
        print("[verify-only] no figure written")
        return
    if not complete and os.environ.get("CAS_ALLOW_PARTIAL", "") != "1":
        print("[refused] cache incomplete; rerun when the experiment finishes, "
              "or set CAS_ALLOW_PARTIAL=1 to plot the partial grid anyway")
        return

    all_data = {b: _aggregate_cells(cells, b) for b in benchmarks_present}
    N_max_count = max(len(d) for d in all_data.values())

    n_rows = len(benchmarks_present)
    fig = plt.figure(figsize=(2.6 * N_max_count + 1.0, 3.0 * n_rows + 0.5))
    gs = fig.add_gridspec(
        n_rows, 2 * N_max_count,
        width_ratios=[3.0, 1.0] * N_max_count,
        wspace=0.05, hspace=0.45,
    )

    for row, bench in enumerate(benchmarks_present):
        data = all_data[bench]
        N_values = sorted(data.keys())

        all_rand = np.concatenate([data[N]["rand"] for N in N_values])
        y_lo = -0.05
        y_hi = max(np.percentile(all_rand, 99), 0.5) * 1.05

        for col_idx, N in enumerate(N_values):
            ax = fig.add_subplot(gs[row, 2 * col_idx])
            ax_d = fig.add_subplot(gs[row, 2 * col_idx + 1], sharey=ax)

            oracle = data[N]["oracle"]
            cas    = data[N]["cas"]
            pca    = data[N]["pca"]
            rand   = data[N]["rand"]
            oracle_med = float(np.median(oracle)) if len(oracle) else 0.0

            # Use a stable hash for jitter rng (Python's built-in hash is
            # randomized across processes; this needs to be reproducible).
            stable_key = abs(hash(f"{bench}_{N}")) % (2 ** 32) \
                if False else (
                    int.from_bytes(
                        f"{bench}_{N}".encode("utf-8"), "little"
                    ) % (2 ** 32)
                )
            rng = np.random.default_rng(stable_key)

            jit_r = rng.uniform(-0.32, 0.32, size=rand.shape)
            ax.scatter(jit_r, rand, s=8, alpha=0.18,
                       color=COLOR_RANDOM, linewidths=0, zorder=2)
            if len(pca):
                jit_p = rng.uniform(-0.14, 0.14, size=pca.shape)
                ax.scatter(np.full_like(pca, 1.0) + jit_p, pca,
                           s=42, alpha=0.95, color=COLOR_PCA,
                           edgecolor="black", linewidth=0.4, zorder=4)
            jit_c = rng.uniform(-0.14, 0.14, size=cas.shape)
            ax.scatter(np.full_like(cas, 2.0) + jit_c, cas,
                       s=42, alpha=0.95, color=COLOR_CAS,
                       edgecolor="black", linewidth=0.4, zorder=4)

            ax.plot([-0.35, 0.35], [np.median(rand)] * 2,
                    color=COLOR_RANDOM, lw=1.4, alpha=0.9, zorder=3)
            if len(pca):
                ax.plot([0.65, 1.35], [np.median(pca)] * 2,
                        color=COLOR_PCA, lw=1.8, alpha=0.95, zorder=5)
            ax.plot([1.65, 2.35], [np.median(cas)] * 2,
                    color=COLOR_CAS, lw=1.8, alpha=0.95, zorder=5)

            ax.axhline(oracle_med, color="0.2", linestyle="--", lw=0.8,
                       alpha=0.6, zorder=1)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Random", "PCA", "CAS"])
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(axis="x", which="both", length=0)
            ax.set_title(rf"$N = {N}$", pad=4)
            if col_idx == 0:
                ax.set_ylabel(rf"{BENCHMARK_TITLES[bench]}"
                              "\n"
                              r"$D_{\mathrm{KL}}(\pi_Z \,\|\, \pi_Z(\cdot;\boldsymbol{V}_r))$  $\downarrow$")
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)

            try:
                kde = gaussian_kde(rand)
                yy = np.linspace(y_lo, y_hi, 400)
                density = kde(yy)
                density /= density.max()
                ax_d.fill_betweenx(yy, 0, density, color=COLOR_RANDOM,
                                    alpha=0.35, linewidth=0)
                ax_d.plot(density, yy, color=COLOR_RANDOM, lw=1.0, alpha=0.9)
            except Exception:
                pass
            ax_d.set_xlim(0, 1.05)
            ax_d.set_xticks([])
            ax_d.tick_params(axis="y", labelleft=False, length=0)
            for spine in ("left", "bottom"):
                ax_d.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    out_path = os.environ.get("RANDOM_SUBSPACE_OUT",
                               os.path.join(HERE, "random_subspace.png"))
    main(out_path=out_path)
