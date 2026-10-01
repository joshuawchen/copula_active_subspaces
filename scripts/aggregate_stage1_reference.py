"""Rebuilds out/{benchmark}_production/cor44_full_decomposition.json from the
tracked cor44 caches (no Monte Carlo) for the Part II supplement figures.

Usage: python3 scripts/aggregate_stage1_reference.py [--benchmark banana]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from cas.stage1_reference import Config            # noqa: E402
from experiments.exp_stage1_reference import build_summary  # noqa: E402

CACHE = os.path.join(REPO, "cache")
OUT = os.path.join(REPO, "out")

# benchmark -> (d_obs, r)
BENCH = {"banana": (20, 4), "even_fold": (20, 4), "conformal_cube": (20, 4)}
K, Q, N_REF = 4, 2, 1_000_000


def _load(name: str):
    with open(os.path.join(CACHE, name)) as f:
        return json.load(f)


def aggregate(b: str) -> str:
    d_obs, r = BENCH[b]
    ref = _load(f"cor44_{b}_ref_K{K}_q{Q}_N{N_REF}.json")
    sat = _load(f"cor44_{b}_saturation_production_N{N_REF}.json")
    scan = _load(f"cor44_{b}_scan_K{K}_q{Q}_r{r}.json")
    disint = _load(f"cor44_{b}_disintegration_K{K}_q{Q}_r{r}.json")

    cfg = Config(d_obs=d_obs, r=r)
    cfg.K_outer, cfg.q_outer = K, Q

    summary = build_summary(cfg, ref, sat, scan, disint)

    out_dir = os.path.join(OUT, f"{b}_production")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cor44_full_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    n_true = sum(1 for row in summary["per_N"] if "true_kl_mean" in row)
    print(f"  {b:<13} -> {os.path.relpath(out_path, REPO)} "
          f"({len(summary['per_N'])} N-rows, {n_true} with measured KL)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=list(BENCH), default=None,
                    help="default: all three")
    args = ap.parse_args()
    benches = [args.benchmark] if args.benchmark else list(BENCH)
    print("Re-aggregating cor44 summaries from committed caches (no MC):")
    for b in benches:
        try:
            aggregate(b)
        except FileNotFoundError as e:
            print(f"  {b:<13} SKIP (missing cache: {os.path.basename(str(e).split(chr(39))[1]) if chr(39) in str(e) else e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
