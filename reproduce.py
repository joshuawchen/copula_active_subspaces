#!/usr/bin/env python3
"""Reproduce the figures and tables of Copula Active Subspaces I and II.

    python reproduce.py                    redraw every figure from the saved results
    python reproduce.py --list             list what produces each figure and table
    python reproduce.py --recompute ITEM   recompute the saved results of one item
                                           (see --list), then redraw the figures
    python reproduce.py --recompute all    recompute every saved result, then redraw
    python reproduce.py --pdf              compile the two papers and their
                                           supplementary materials
    python reproduce.py --check-numbers    recompute every number reported in the
                                           text and tables from the saved results
                                           and compare it with the LaTeX sources

Recomputation skips results that are already saved in cache/; delete a result
file to recompute it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
EXAMPLES = ("banana", "even_fold", "conformal_cube")   # Examples 1, 2 and 3
PART2_SEEDS = "10"
DECOMPOSITION_WORKERS = "2"    # the decomposition's processes hold the most memory

# Figure files the LaTeX sources include, and where the figure scripts write them.
FIGS_FROM_FIGURES_DIR = ["random_subspace.png", "rate_plot.png"]
FIGS_FROM_OUT_DIR = ["pedagogical_2d.png", "recovery_row.png", "test_ll_all.png",
                     "bip_kl_all.png", "bip_posteriors.png", "eigvals_all.png",
                     "fig_composite_main_banana.pdf", "figSy_v2_even_fold.pdf",
                     "figSy_v2_conformal_cube.pdf"]

DRAWN_DIRECTLY = [
    ("Part I, Figure 1 (also Part II supplement, Figure SM1)", "figures/fig01_pedagogical.py"),
    ("Part I supplement, Figure SM1", "figures/figS_eigvals.py"),
    ("Part II supplement, Figure SM2", "figures/figS1_rate.py"),
]


def ex(script, *args):
    return [PY, os.path.join("experiments", script), *args]


def items(workers):
    """(name, what it produces, commands), in the order --recompute all runs
    them; later Part II items read the saved results of earlier ones."""
    reference = [ex("exp_stage1_reference.py", "--benchmark", b,
                    "--K_outer", "4", "--q_outer", "2", "--n_seeds", PART2_SEEDS,
                    "--n_workers", workers, "--disint_n_workers", workers,
                    "--inner-sampler", "quad") for b in EXAMPLES]
    decomposition = [ex("exp_stage2_oracle_kl.py", "--benchmark", b,
                        "--K_outer", "4", "--q_outer", "2", "--n_seeds", PART2_SEEDS,
                        "--n_workers", workers,
                        "--decomp_n_workers", DECOMPOSITION_WORKERS) for b in EXAMPLES]
    return [
        ("part1-main", "Part I, Figures 2-4 and the results of Section 5", [
            ex("exp_noise_entropy.py"),
            ex("exp_section51_subspace.py"),
            [PY, "-m", "experiments.exp_paper_run"]]),
        ("part1-posteriors", "Part I, Figure 5", [
            ex("exp_section53_hero.py", "--benchmark", b) for b in EXAMPLES]),
        ("part1-normalizer", "Part I, normalizer accuracy reported in the appendix", [
            [PY, "-m", "experiments.exp_normalizer_stability"]]),
        ("part1-pca", "Part I supplement, Tables SM1 and SM2", [
            ex("exp_sm1_pca_ablation.py", "--stage", "all", "--n-seeds", "20",
               "--n-trials", "30")]),
        ("part1-random-subspaces", "Part I supplement, Figure SM2", [
            ex("exp_random_subspace_oracle_kl.py", "--n-workers", workers)]),
        ("part1-tail-sum", "Part I supplement, Table SM3", [
            ex("exp_ehat_tightness.py", "--benchmark", "all")]),
        ("part1-regularizer", "Part I supplement, Tables SM4-SM7", [
            ex("exp_sm3_c_stability.py"),
            ex("exp_sm3_rh2_kappa.py"),
            *[ex("exp_regularizer_ablation.py", "--benchmark", b) for b in EXAMPLES]]),
        ("part1-centering", "Part I supplement, Tables SM8 and SM9", [
            *[ex("exp_centering_ablation.py", "--benchmark", b) for b in EXAMPLES],
            *[ex("exp_k2_sweep_centering.py", "--benchmark", b) for b in EXAMPLES]]),
        ("part2-reference", "Part II, reference Stage-1 quantities used by Figures 1-3",
         reference),
        ("part2-decomposition", "Part II, Figure 1 and supplement Figures SM3 and SM4", [
            ex("exp_delta_sub_Z.py", "--benchmark", "all", "--cells",
               "--n_outer", "4000", "--nq", "7"),
            *decomposition]),
        ("part2-a-posteriori", "Part II, Figures 2 and 3", [
            ex("exp_stage1_aposteriori.py", "--benchmark", "all"),
            *[ex("exp_apost_endtoend.py", "--benchmark", b) for b in EXAMPLES]]),
        ("part2-truncation", "Part II, estimate of the truncation term reported in the text", [
            ex("exp_hier_trunc.py", "--out", "cache", "--enrich", "q"),
            ex("exp_hier_trunc.py", "--out", "cache", "--enrich", "K")]),
        ("part2-certificate", "Part II, Section 5, ratio of the bound to the divergence at the estimated subspace", [
            ex("exp_stage1_cert.py", "--benchmark", "all"),
            ex("exp_delta_sub_Z.py", "--benchmark", "all", "--n_outer", "4000", "--nq", "11"),
            ex("exp_delta_sub_Z.py", "--benchmark", "all", "--cells",
               "--n_outer", "4000", "--nq", "7")]),
        ("part2-sharper-bound", "Part II supplement, Table SM1", [
            ex("exp_jhi_supplement.py", "--stage", "all")]),
    ]


def run(cmd, env):
    shown = ["python" if c == PY else c for c in cmd]
    print(f"$ {' '.join(shown)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, env=env)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(shown)}")


def draw_figures(env):
    run([PY, os.path.join("figures", "make_all.py")], env)
    run([PY, os.path.join("figures", "fig_stage1_bounds.py"), "--benchmark", "banana",
         "--out", os.path.join("paper", "fig_stage1_bounds")], env)
    run([PY, os.path.join("figures", "fig_apost_endtoend.py"),
         "--out", os.path.join("paper", "fig_apost_endtoend")], env)
    for folder, names in (("figures", FIGS_FROM_FIGURES_DIR), ("out", FIGS_FROM_OUT_DIR)):
        for name in names:
            src = os.path.join(HERE, folder, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(HERE, "paper", name))
            else:
                print(f"  not found: {folder}/{name}")
    print("figures written to out/ and copied to paper/")


def build_pdfs():
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex not found")
    docs = ["part1_method", "part1_supplement", "part2_analysis", "part2_supplement"]
    for _ in range(3):   # three passes resolve the references between each paper and its supplement
        for d in docs:
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                f"{d}.tex"], cwd=os.path.join(HERE, "paper"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                raise SystemExit(f"pdflatex failed on {d}.tex; see paper/{d}.log")
    print("compiled " + ", ".join(f"paper/{d}.pdf" for d in docs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true",
                   help="list what produces each figure and table")
    g.add_argument("--recompute", metavar="ITEM",
                   help="recompute the saved results of ITEM, or of all items")
    g.add_argument("--pdf", action="store_true",
                   help="compile the two papers and their supplementary materials")
    g.add_argument("--check-numbers", action="store_true",
                   help="compare every reported number with the saved results")
    ap.add_argument("--workers", default="4",
                    help="parallel processes for the long experiments (default 4)")
    a = ap.parse_args()

    table = items(a.workers)
    if a.list:
        print("Recomputable items (python reproduce.py --recompute ITEM):")
        for name, what, _ in table:
            print(f"  {name:24s} {what}")
        print("\nFigures computed directly when drawn:")
        for what, script in DRAWN_DIRECTLY:
            print(f"  {what}: {script}")
        return 0
    if a.pdf:
        build_pdfs()
        return 0
    if a.check_numbers:
        return subprocess.run([PY, os.path.join("scripts", "check_paper_numbers.py")],
                              cwd=HERE).returncode

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(HERE, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["CAS_N_WORKERS"] = a.workers
    env["SCALAR_MARGINAL_WORKERS"] = "1"
    env.setdefault("N_WORKERS", a.workers)

    if a.recompute:
        names = [t[0] for t in table]
        if a.recompute != "all" and a.recompute not in names:
            raise SystemExit(f"unknown item {a.recompute!r}; see python reproduce.py --list")
        for name, what, cmds in table:
            if a.recompute in ("all", name):
                print(f"\n=== {name}: {what} ===", flush=True)
                for cmd in cmds:
                    run(cmd, env)
    draw_figures(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
