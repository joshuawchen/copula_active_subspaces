"""Part II supplement: composite KL decomposition for Examples 2 and 3.
Reads out/exp_stage2_oracle_kl.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


# The colours of fig_composite_main_banana (main-text Figure 1) and
# fig_apost_endtoend, so a term has the same colour in every figure of Part II.
C_SUB = "#2a78d6"
C_MARG = "#eda100"
C_DEN = "#1baf7a"
C_TOT = "#0b0b0b"
C_SUM = "#6b6b6b"
REF = r"$\mathcal{D}_{\mathrm{KL}}(\pi_n\|\hat\pi_n)$"


def _plot(summary: dict, out_path: str, benchmark: str) -> None:
    # The paper refers to the three problems as Examples 1-3; the benchmark
    # keys are internal names and must not reach a printed panel title.
    EXAMPLE = {"banana": "Example 1", "even_fold": "Example 2",
               "conformal_cube": "Example 3"}
    label = EXAMPLE.get(benchmark, benchmark)
    apply_paper_style()
    rows = summary["per_N"]
    Ns = np.array([r["N"] for r in rows])

    full_m = np.array([r["full_kl_mean"] for r in rows])
    full_s = np.array([r["full_kl_std"] for r in rows])
    dm_m = np.array([r["delta_marg_mean"] for r in rows])
    dm_s = np.array([r["delta_marg_std"] for r in rows])
    dd_m = np.array([r["delta_den_mean"] for r in rows])
    dd_s = np.array([r["delta_den_std"] for r in rows])
    ds_m = np.array([r.get("delta_sub_mean", np.nan) for r in rows])
    ds_s = np.array([r.get("delta_sub_std", np.nan) for r in rows])

    have_sub = not np.all(np.isnan(ds_m))
    if have_sub:
        bound_m = dm_m + ds_m + dd_m
    else:
        bound_m = dm_m + dd_m

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: stacked contributions vs N
    ax = axes[0]
    x = np.arange(len(Ns))
    width = 0.7
    # Clip negative values to 0 for stacking (Δ_den can be slightly negative
    # under MC noise; for the figure we still show the stacking).
    dm_pos = np.clip(dm_m, 0, None)
    dd_pos = np.clip(dd_m, 0, None)
    bars = []
    bars.append(ax.bar(x, dm_pos, width, label=r"$\mathcal{E}_{\mathrm{marg}}$",
                          color=C_MARG))
    bottom = dm_pos.copy()
    if have_sub:
        ds_pos = np.clip(ds_m, 0, None)
        bars.append(ax.bar(x, ds_pos, width, bottom=bottom,
                              label=r"$\mathcal{E}_{\mathrm{proj}}$", color=C_SUB))
        bottom = bottom + ds_pos
    bars.append(ax.bar(x, dd_pos, width, bottom=bottom,
                          label=r"$\mathcal{E}_{\mathrm{dens}}$", color=C_DEN))
    # Overlay the reference total as a marker
    ax.errorbar(x, full_m, yerr=full_s, fmt="o", color=C_TOT, ms=7,
                  capsize=3, label="reference " + REF,
                  zorder=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in Ns], rotation=30)
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel("KL (nats)")
    ax.set_title(f"(A) decomposition, {label}")
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: three-term sum against the reference total (log y)
    ax = axes[1]
    ax.errorbar(Ns, bound_m, color=C_SUM, ls="--", lw=2, marker="s", ms=6,
                  label=r"$\mathcal{E}_{\mathrm{tot}}$ (three-term sum)")
    ax.errorbar(Ns, full_m, yerr=full_s, color=C_TOT, lw=2, marker="o", ms=6,
                  label="reference " + REF)
    ax.set_xscale("log")
    # Guard against negative full_KL for log scale
    if (full_m > 0).all() and (bound_m > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("KL (nats)")
    ax.set_title(f"(B) three-term sum against reference, {label}")
    ax.legend(fontsize=9, loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # Panel C: per-term log-log convergence
    ax = axes[2]
    ax.errorbar(Ns, np.abs(dm_m), yerr=dm_s, color=C_MARG, lw=1.8,
                  marker="o", ms=5, label=r"$|\mathcal{E}_{\mathrm{marg}}|$")
    if have_sub:
        ax.errorbar(Ns, np.abs(ds_m), yerr=ds_s, color=C_SUB, lw=1.8,
                      marker="^", ms=5, label=r"$|\mathcal{E}_{\mathrm{proj}}|$")
    ax.errorbar(Ns, np.abs(dd_m), yerr=dd_s, color=C_DEN, lw=1.8,
                  marker="s", ms=5, label=r"$|\mathcal{E}_{\mathrm{dens}}|$")
    ax.errorbar(Ns, np.abs(full_m), yerr=full_s, color=C_TOT, lw=2.2,
                  marker="D", ms=6, label=r"$|\mathcal{D}_{\mathrm{KL}}(\pi_n\|\hat\pi_n)|$")
    ax.set_xscale("log")
    if (full_m > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("KL (nats)")
    ax.set_title(f"(C) per-term convergence, {label}")
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_path", type=str, default=None,
                   help="Path to exp_stage2_oracle_kl.json. Defaults to "
                        "out/{benchmark}_production/exp_stage2_oracle_kl.json")
    p.add_argument("--benchmark", type=str, default="banana",
                   choices=["banana", "even_fold", "conformal_cube"])
    p.add_argument("--out", type=str, default=None,
                   help="Output path; defaults to figures/figSy_v2_{benchmark}.pdf")
    args = p.parse_args()

    if args.summary_path is None:
        candidates = [
            os.path.join(_REPO, "out", f"{args.benchmark}_production",
                         "exp_stage2_oracle_kl.json"),
            os.path.join(_REPO, "out", "exp_stage2_oracle_kl.json"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                args.summary_path = cand
                break
        else:
            print(f"ERROR: no summary at any of {candidates}. "
                  f"Run exp_stage2_oracle_kl.py first.", file=sys.stderr)
            sys.exit(1)

    if args.out is None:
        args.out = os.path.join(HERE, f"figSy_v2_{args.benchmark}.pdf")

    with open(args.summary_path) as f:
        summary = json.load(f)

    _plot(summary, args.out, args.benchmark)


if __name__ == "__main__":
    main()
