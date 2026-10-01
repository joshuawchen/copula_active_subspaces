"""
make_all.py — Build every figure of the two papers in one command.

Runs each figure script under ``figures/`` in a fresh Python process,
writing all outputs to ``OUT_DIR`` (default: ``out/`` at repo root,
override with the ``CAS_FIGURES_OUT_DIR`` env var). reproduce.py then
copies the files the documents include into ``paper/``.

Usage:
    python figures/make_all.py
    CAS_FIGURES_OUT_DIR=/some/dir python figures/make_all.py
    N_WORKERS=8 python figures/make_all.py             # parallelism for figS1_rate

fig_stage1_bounds.py and fig_apost_endtoend.py (Part II) are run by
reproduce.py directly, since they write to paper/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(REPO, "out")
OUT_DIR = os.environ.get("CAS_FIGURES_OUT_DIR", DEFAULT_OUT)
os.makedirs(OUT_DIR, exist_ok=True)


# Each row: (script, args, env_var_for_output_path | None, label,
#            output filenames the script produces). `args` is a list of
#            extra CLI args (may contain the literal token OUT_DIR, replaced
#            with the resolved output directory at run time).
BENCHES = ["banana", "even_fold", "conformal_cube"]

JOBS = [
    # Part I main text
    ("fig01_pedagogical.py",         [], "PEDAGOGICAL_OUT",
     "Part I  Fig 1  pedagogical_2d",  ["pedagogical_2d.png"]),

    ("fig03_recovery_row.py", ["--out", "OUT_DIR/recovery_row.png"], None,
     "Part I  Fig 2  recovery_row",    ["recovery_row.png"]),

    ("fig_kl_all_examples.py",
     ["--which", "testll", "--out", "OUT_DIR/test_ll_all.png"], None,
     "Part I  Fig 3  test_ll_all",     ["test_ll_all.png"]),

    ("fig_kl_all_examples.py",
     ["--which", "bip", "--out", "OUT_DIR/bip_kl_all.png"], None,
     "Part I  Fig 4  bip_kl_all",      ["bip_kl_all.png"]),

    ("fig05_bip_posteriors.py",
     ["--benchmark", "banana", "--out", "OUT_DIR/bip_posteriors.png"], None,
     "Part I  Fig 5  bip_posteriors",  ["bip_posteriors.png"]),

    # Part I supplement
    ("figS_eigvals.py",              ["--all", "--out-dir", "OUT_DIR"], None,
     "Part I  SM     eigvals_all",     ["eigvals_all.png"]),

    ("figS3_random_subspace.py",     [], None,
     "Part I  SM     random_subspace", ["random_subspace.png"]),

    # Part II main text (fig_stage1_bounds and fig_apost_endtoend are run by
    # reproduce.py directly)
    ("fig_composite_main_banana.py",
     ["--benchmark", "banana", "--save_dir", "OUT_DIR"], None,
     "Part II Fig    composite_main_banana", ["fig_composite_main_banana.pdf",
                                             "fig_composite_main_banana.png"]),

    # Part II supplement
    ("figS1_rate.py",                [], None,
     "Part II SM     rate_plot",       ["rate_plot.png"]),

    *[("figSy_composite_bound_v2.py",
       ["--benchmark", b, "--out", f"OUT_DIR/figSy_v2_{b}.pdf"], None,
       f"Part II SM     figSy_v2_{b}", [f"figSy_v2_{b}.pdf", f"figSy_v2_{b}.png"])
      for b in ("even_fold", "conformal_cube")],
]


def _resolve_env(out_var: str | None) -> str | None:
    """Compute the value to set the env var to (file path or directory)."""
    if out_var is None:
        return None
    if out_var.endswith("_DIR"):
        return OUT_DIR
    # Single-file env var: look up the first output filename for the script
    for _, _, var, _, files in JOBS:
        if var == out_var:
            return os.path.join(OUT_DIR, files[0])
    return OUT_DIR


def _run_one(script: str, args: list[str], out_var: str | None,
             label: str) -> tuple[str, float]:
    script_path = os.path.join(HERE, script)
    if not os.path.exists(script_path):
        print(f"  [skip] {label}: script {script} missing")
        return "skip", 0.0

    env = os.environ.copy()
    if out_var is not None:
        env[out_var] = _resolve_env(out_var) or ""

    # Resolve the OUT_DIR token in any CLI args (e.g. --out OUT_DIR/foo.pdf).
    resolved_args = [a.replace("OUT_DIR", OUT_DIR) for a in args]

    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, script_path, *resolved_args],
                           env=env, cwd=OUT_DIR,
                           capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"  [FAIL] {label} ({elapsed:.1f}s)")
            print(f"    stdout: {r.stdout[-300:]}")
            print(f"    stderr: {r.stderr[-300:]}")
            return "fail", elapsed
        tail = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        # A script may print "[skip]" when it exits cleanly without producing
        # its output (missing cache, etc.). Surface that as a SKIP rather than
        # a silent OK so the user notices.
        if "[skip]" in r.stdout:
            print(f"  [SKIP] {label:<32s} ({elapsed:>5.1f}s)  {tail[:80]}")
            return "skip", elapsed
        print(f"  [ok]   {label:<32s} ({elapsed:>5.1f}s)  {tail[:80]}")
        return "ok", elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {label} after {elapsed:.0f}s")
        return "fail", elapsed


def main() -> int:
    print("=" * 72)
    print("Building all paper figures")
    print(f"  output dir : {OUT_DIR}")
    print(f"  N_WORKERS  : {os.environ.get('N_WORKERS', 'auto')}")
    print("=" * 72)
    print()

    n_ok = n_skip = n_fail = 0
    skipped_labels = []
    total = 0.0
    for script, args, var, label, _ in JOBS:
        status, elapsed = _run_one(script, args, var, label)
        total += elapsed
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
            skipped_labels.append(label)
        else:
            n_fail += 1

    print()
    print("=" * 72)
    print(f"Summary:  {n_ok}/{len(JOBS)} jobs succeeded   ({total:.0f}s total)")
    if n_skip:
        print(f"          {n_skip} skipped (missing cache — see below)")
        for lbl in skipped_labels:
            print(f"            - {lbl}")
        print("          Hint: regenerate the missing cache with the reproduce.py stage that writes it.")
    if n_fail:
        print(f"          {n_fail} failed (see above)")
    print(f"Outputs in: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        if os.path.isfile(path):
            print(f"  {f:<40s}  {os.path.getsize(path):>10,} bytes")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
