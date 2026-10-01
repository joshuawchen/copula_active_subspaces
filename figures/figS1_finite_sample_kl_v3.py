"""Part II supplement: Stage-1 bound and measured Stage-1 KL versus N.
Reads out/{benchmark}_production/cor44_full_decomposition.json.
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

OUT_DIR_DEFAULT = os.path.join(_REPO, "out")


def _plot(summary: dict, out_path: str, benchmark: str) -> None:
    apply_paper_style()
    ref = summary["reference"]
    E_r = float(ref["E_r"])
    T2_sq = float(ref.get("basis_truncation_sq", 0.0))
    sqrt_Er = float(np.sqrt(max(E_r, 0.0)))
    hermite_gap = float(np.sqrt(max(T2_sq, 0.0)))  # ||S - S_Lambda||

    rows = summary["per_N"]
    Ns = np.array([r["N"] for r in rows])
    sts_m = np.array([r["sin_theta_mean"] for r in rows])
    sts_s = np.array([r["sin_theta_std"] for r in rows])

    # Leftover parameterized variance Tr((I - Phat_r) C_Lambda) = 2 * lsi_bound,
    # and the Stage-1 bound 0.5 (sqrt(leftover) + ||S - S_Lambda||)^2, per seed.
    leftover_sqrt_m, leftover_sqrt_s = [], []
    rhs_m, rhs_s = [], []
    for row in rows:
        lsi_ps = np.asarray(row["lsi_bound_per_seed"], dtype=float)
        leftover_ps = np.sqrt(np.maximum(2.0 * lsi_ps, 0.0))   # sqrt(Tr((I-P)C_L))
        rhs_ps = 0.5 * (leftover_ps + hermite_gap) ** 2        # Stage-1 bound
        leftover_sqrt_m.append(leftover_ps.mean())
        leftover_sqrt_s.append(leftover_ps.std())
        rhs_m.append(rhs_ps.mean())
        rhs_s.append(rhs_ps.std())
    leftover_sqrt_m = np.array(leftover_sqrt_m)
    leftover_sqrt_s = np.array(leftover_sqrt_s)
    rhs_m = np.array(rhs_m)
    rhs_s = np.array(rhs_s)

    # Measured Stage-1 KL via conditional Monte Carlo. Missing -> NaN (line breaks).
    true_m = np.array([r.get("true_kl_mean", np.nan) for r in rows])
    true_s = np.array([r.get("true_kl_std", np.nan) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: decomposition of the Stage-1 bound (eq:stage1-decomp)
    ax = axes[0]
    ax.errorbar(Ns, leftover_sqrt_m, yerr=leftover_sqrt_s, color="C2", lw=1.8,
                marker="o", ms=5,
                label=r"$\sqrt{\mathrm{Tr}((I-\hat P_r)C_\Lambda)}$ (leftover var.)")
    ax.axhline(sqrt_Er, color="C0", ls="-", lw=1.8,
               label=f"$\\sqrt{{E_r(C_\\Lambda)}} = {sqrt_Er:.2f}$ (rank floor)")
    ax.axhline(hermite_gap, color="C1", ls="-", lw=1.8,
               label=f"$\\|\\mathcal{{S}}-\\mathcal{{S}}_\\Lambda\\| = {hermite_gap:.2f}$ (Hermite gap)")
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (training samples)")
    ax.set_ylabel("Term value (nats$^{1/2}$)")
    ax.set_title(f"(A) Bound decomposition [{benchmark}]")
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # Panel B: bound vs measured Stage-1 KL
    ax = axes[1]
    ax.errorbar(Ns, rhs_m, yerr=rhs_s, color="C3", ls="--", lw=2.0,
                marker="s", ms=6, label="Stage-1 bound (Thm 4.2)")
    if not np.all(np.isnan(true_m)):
        valid = ~np.isnan(true_m)
        ax.errorbar(Ns[valid], true_m[valid], yerr=true_s[valid],
                    color="C0", ls="-", lw=2.0, marker="o", ms=6,
                    label="measured Stage-1 KL (disint. MC)")
    ax.axhline(0.5 * E_r, color="gray", ls=":", alpha=0.6,
               label=f"$\\tfrac{{1}}{{2}} E_r = {0.5 * E_r:.3f}$ (floor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ (training samples)")
    ax.set_ylabel("KL (nats)")
    ax.set_title(f"(B) Stage-1 bound vs measured [{benchmark}]")
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # Panel C: sin Theta rate verification
    ax = axes[2]
    ax.errorbar(Ns, sts_m, yerr=sts_s, color="C2", lw=2.0, marker="o", ms=6,
                label=r"$\|\sin\Theta(\hat V_r, V_{r,\Lambda})\|_F$")
    ref_Ns = np.array(Ns, dtype=float)
    if len(Ns) > 1:
        sts_pos = sts_m[~np.isnan(sts_m)]
        if len(sts_pos) > 0:
            ref_y0 = sts_m[-1] * np.sqrt(Ns[-1])
            ref_y = ref_y0 / np.sqrt(ref_Ns)
            ax.plot(ref_Ns, ref_y, color="gray", ls="--", lw=1.0,
                    label=r"$\mathcal{O}(N^{-1/2})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ (training samples)")
    ax.set_ylabel(r"$\|\sin\Theta\|_F$")
    ax.set_title(f"(C) Subspace rate [{benchmark}]")
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
                   help="Path to cor44_full_decomposition.json. Defaults to "
                        "out/{benchmark}_production/cor44_full_decomposition.json.")
    p.add_argument("--benchmark", type=str, default="banana",
                   choices=["banana", "even_fold", "conformal_cube",
                            "cubic_banana", "ppg"],
                   help="for default summary_path / out path / title")
    p.add_argument("--out", type=str, default=None,
                   help="Output PDF path. Defaults to figures/figS1_v3_{benchmark}.pdf")
    args = p.parse_args()

    if args.summary_path is None:
        candidates = [
            os.path.join(_REPO, "out", f"{args.benchmark}_production",
                         "cor44_full_decomposition.json"),
            os.path.join(_REPO, "out", "cor44_full_decomposition.json"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                args.summary_path = cand
                break
        else:
            print(f"ERROR: no summary found at any of {candidates}. "
                  f"Run exp_stage1_reference.py first.",
                  file=sys.stderr)
            sys.exit(1)

    if args.out is None:
        args.out = os.path.join(HERE, f"figS1_v3_{args.benchmark}.pdf")

    with open(args.summary_path) as f:
        summary = json.load(f)

    _plot(summary, args.out, args.benchmark)


if __name__ == "__main__":
    main()
