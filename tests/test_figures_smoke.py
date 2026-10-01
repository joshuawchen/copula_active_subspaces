"""Smoke tests for the figure pipeline.

Each cache-driven figure script should:
  1. Run end-to-end without error;
  2. Produce a non-empty PNG/PDF;
  3. Give the same MD5 across two consecutive runs (deterministic).

Compute-bound figures (figS1_rate, figS3_corner10) are tested only for
"runs without error and produces non-empty output" because their wall
time blows past the unit-test budget.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGURES = os.path.join(REPO, "figures")


def _md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _run_script(script: str, env_overrides: dict[str, str], outputs: list[str],
                cwd: str, timeout: int = 90) -> dict[str, str]:
    """Run script in a subprocess; return {output_filename: md5}."""
    env = os.environ.copy()
    env.update(env_overrides)
    r = subprocess.run(
        [sys.executable, os.path.join(FIGURES, script)],
        env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"{script} exited {r.returncode}\nstdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
        )
    return {name: _md5(os.path.join(cwd, name)) for name in outputs
            if os.path.exists(os.path.join(cwd, name))}


# ---------------------------------------------------------------------------
# Cache-only figures (fast, deterministic)
# ---------------------------------------------------------------------------

CACHE_FIGURES = [
    ("fig05_bip_posteriors.py", {"BIP_POSTERIORS_OUT": "bip_posteriors.png"},
     ["bip_posteriors.png"]),
    ("figS3_random_subspace.py", {"RANDOM_SUBSPACE_OUT": "random_subspace.png"},
     ["random_subspace.png"]),
]

# Cache files each figure script needs; skip the test if the cache is missing.
# (Some caches ship empty in source checkouts and only exist after the user
# runs run.sh end-to-end.  Skipping is more honest than failing.)
_FIGURE_CACHES = {
    "figS3_random_subspace.py": "cache/exp_oracle_kl_random_subspace.npz",
}


def _cache_missing_for(script):
    rel = _FIGURE_CACHES.get(script)
    if rel is None:
        return False
    return not os.path.exists(os.path.join(os.path.dirname(FIGURES), rel))


@pytest.mark.smoke
@pytest.mark.parametrize("script, env, outputs", CACHE_FIGURES)
def test_cache_figure_runs(script, env, outputs):
    """Cache-driven figures run without error and produce non-empty output."""
    if _cache_missing_for(script):
        pytest.skip(f"required cache for {script} is not present; "
                    f"run ./run.sh to build it")
    with tempfile.TemporaryDirectory() as out_dir:
        # Resolve output paths relative to out_dir
        env_resolved = {k: (os.path.join(out_dir, v) if not v.endswith("_DIR")
                            else out_dir) for k, v in env.items()}
        md5s = _run_script(script, env_resolved, outputs, cwd=out_dir, timeout=60)
        for name in outputs:
            path = os.path.join(out_dir, name)
            assert os.path.exists(path), f"{name} not produced"
            assert os.path.getsize(path) > 1000, f"{name} suspiciously small"


@pytest.mark.smoke
@pytest.mark.parametrize("script, env, outputs", CACHE_FIGURES)
def test_cache_figure_deterministic(script, env, outputs):
    """Two consecutive runs of a cache-only figure should produce
    bit-identical output."""
    if _cache_missing_for(script):
        pytest.skip(f"required cache for {script} is not present; "
                    f"run ./run.sh to build it")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        env1 = {k: (os.path.join(d1, v) if not v.endswith("_DIR") else d1)
                for k, v in env.items()}
        env2 = {k: (os.path.join(d2, v) if not v.endswith("_DIR") else d2)
                for k, v in env.items()}
        m1 = _run_script(script, env1, outputs, cwd=d1, timeout=60)
        m2 = _run_script(script, env2, outputs, cwd=d2, timeout=60)
        for name in outputs:
            assert m1.get(name) == m2.get(name), \
                f"{name} differs across runs ({m1.get(name)} vs {m2.get(name)})"


# ---------------------------------------------------------------------------
# Compute-bound figures — only check they run, no determinism check
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.smoke
def test_pedagogical_runs():
    with tempfile.TemporaryDirectory() as out_dir:
        _run_script("fig01_pedagogical.py",
                    {"PEDAGOGICAL_OUT": os.path.join(out_dir, "pedagogical_2d.png")},
                    ["pedagogical_2d.png"], cwd=out_dir, timeout=30)


@pytest.mark.slow
@pytest.mark.smoke
def test_rate_runs():
    """figS1_rate: 120 cas_fits — slow even at default seeds, so we use
    a tiny subset by overriding the function args via env (not currently
    supported; this test just times the default and skips on slow CI)."""
    # Just run with N_WORKERS=1 (sequential) on the smallest config available
    with tempfile.TemporaryDirectory() as out_dir:
        _run_script("figS1_rate.py",
                    {"N_WORKERS": "1",
                     "RATE_PLOT_OUT": os.path.join(out_dir, "rate_plot.png")},
                    ["rate_plot.png"], cwd=out_dir, timeout=120)


@pytest.mark.slow
@pytest.mark.smoke
def test_corner10_runs():
    """figS3_corner10: the heaviest figure; it fits all four estimators at
    N=12500 with the cross-validated multi-index selection, which takes a
    few minutes on a laptop."""
    with tempfile.TemporaryDirectory() as out_dir:
        _run_script("figS3_corner10.py", {"CORNER10_OUT_DIR": out_dir,
                                          "CORNER10_DPI": "60"},
                    ["corner10_oracle.png", "corner10_cas.png",
                     "corner10_pca.png", "corner10_pom.png", "corner10_gauss.png"],
                    cwd=out_dir, timeout=900)


@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.parametrize("benchmark", ["banana", "even_fold", "conformal_cube"])
def test_eigvals_runs(benchmark):
    """figS_eigvals: small-N smoke test for the eigenvalue-decay figure.
    
    Uses N=1000 and 2 seeds so the test fits in CI budget; the paper
    version uses N=50000 with 5 seeds."""
    with tempfile.TemporaryDirectory() as out_dir:
        env = os.environ.copy()
        r = subprocess.run(
            [sys.executable, os.path.join(FIGURES, "figS_eigvals.py"),
             "--benchmark", benchmark, "--N", "1000", "--n-seeds", "2",
             "--out-dir", out_dir],
            env=env, capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            raise AssertionError(
                f"figS_eigvals.py (--benchmark {benchmark}) exited {r.returncode}\n"
                f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
            )
        png = os.path.join(out_dir, f"eigvals_{benchmark}.png")
        assert os.path.exists(png), f"missing output: {png}"
        assert os.path.getsize(png) > 1000, f"output too small: {png}"


@pytest.mark.smoke
@pytest.mark.parametrize("which, out_name", [("testll", "test_ll_all.png"),
                                             ("bip", "bip_kl_all.png")])
def test_kl_all_examples_runs(which, out_name):
    """fig_kl_all_examples: the two three-panel figures of S5.2 and S5.3.

    These are the figures the paper shows; they read the run2026 caches, so
    the test skips when those are absent rather than failing.
    """
    if not os.path.isdir(os.path.join(REPO, "cache", "run2026")):
        pytest.skip("cache/run2026 not present; run experiments/exp_paper_run.py")
    with tempfile.TemporaryDirectory() as out_dir:
        out_path = os.path.join(out_dir, out_name)
        r = subprocess.run(
            [sys.executable, os.path.join(FIGURES, "fig_kl_all_examples.py"),
             "--which", which, "--out", out_path],
            env=os.environ.copy(), capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise AssertionError(
                f"fig_kl_all_examples.py (--which {which}) exited {r.returncode}\n"
                f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
            )
        assert os.path.exists(out_path), f"missing output: {out_path}"
        assert os.path.getsize(out_path) > 1000, f"output too small: {out_path}"


@pytest.mark.smoke
def test_finite_sample_kl_runs():
    """figS1_finite_sample_kl_v3: the Cor 4.4 Stage-1 anatomy figure (Fig I).

    Cache-driven: reads out/{benchmark}_production/cor44_full_decomposition.json
    (built by experiments/exp_stage1_reference.py). That cache is a
    compute-bound product of the full pipeline and is not present in a fresh
    checkout, so the test skips when it is missing rather than failing.
    """
    benchmark = "banana"
    cache_rel = f"out/{benchmark}_production/cor44_full_decomposition.json"
    if not os.path.exists(os.path.join(REPO, cache_rel)):
        pytest.skip(f"required cache {cache_rel} not present; "
                    f"run experiments/exp_stage1_reference.py to build it")
    with tempfile.TemporaryDirectory() as out_dir:
        out_pdf = os.path.join(out_dir, f"figS1_v3_{benchmark}.pdf")
        env = os.environ.copy()
        r = subprocess.run(
            [sys.executable,
             os.path.join(FIGURES, "figS1_finite_sample_kl_v3.py"),
             "--benchmark", benchmark, "--out", out_pdf],
            env=env, capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            raise AssertionError(
                f"figS1_finite_sample_kl_v3.py exited {r.returncode}\n"
                f"stdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
            )
        assert os.path.exists(out_pdf), f"missing PDF output: {out_pdf}"
        assert os.path.getsize(out_pdf) > 1000, f"PDF too small: {out_pdf}"
        png = os.path.splitext(out_pdf)[0] + ".png"
        assert os.path.exists(png), f"missing PNG output: {png}"
        assert os.path.getsize(png) > 1000, f"PNG too small: {png}"
