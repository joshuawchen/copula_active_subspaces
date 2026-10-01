"""Streaming Hermite-feature machinery for the Part II reference computations:
chunked assembly of the score-matching system and the score matrix,
memmap-backed marginal distribution functions, and the pool, saturation,
reference and scan stages driven by a Config.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from itertools import combinations
from typing import Callable, Optional

import numpy as np
from scipy import stats

try:
    from threadpoolctl import threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:
    _HAS_THREADPOOLCTL = False

    def threadpool_limits(limits=None, user_api=None):
        """No-op fallback if threadpoolctl not installed."""
        class _NoOp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _NoOp()


# =====================================================================
# System / memory helpers
# =====================================================================

def detect_physical_cores() -> int:
    """Best-effort physical core count. Falls back to os.cpu_count()."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    if sys.platform == 'darwin':
        try:
            import subprocess
            out = subprocess.check_output(
                ['sysctl', '-n', 'hw.perflevel0.physicalcpu']
            )
            return int(out.strip())
        except Exception:
            try:
                out = subprocess.check_output(
                    ['sysctl', '-n', 'hw.physicalcpu']
                )
                return int(out.strip())
            except Exception:
                pass
    return os.cpu_count() or 4


def estimate_peak_gb(K: int, q: int, d: int, N_chunk: int,
                     worker_overhead_gb: float = 0.3) -> float:
    """Estimate peak resident memory for one worker at (K, q).

    Components, all in bytes:
      - Gram matrix |Lambda|^2 float64       : dominant at large |Lambda|
      - 2 axis-Phi  N_chunk * |Lambda| float32 (Phi + Phi^2)
      - Hermite table d * N_chunk * (K+1) float32
      - worker fork overhead (numpy/scipy import + heap)
    """
    A_size = len(enumerate_dictionary(d, K, q))
    gram = A_size ** 2 * 8
    phi = 2 * N_chunk * A_size * 4
    herm = d * N_chunk * (K + 1) * 4
    return (gram + phi + herm) / 1e9 + worker_overhead_gb


# =====================================================================
# Hermite features and dictionary
# =====================================================================

def hermite_norm(z: np.ndarray, max_degree: int) -> np.ndarray:
    """Normalized probabilist Hermite polynomials H_0..H_K, shape (..., K+1).

    Recurrence: H_k(z) = (z H_{k-1}(z) - sqrt(k-1) H_{k-2}(z)) / sqrt(k).
    Orthonormal in L^2(gamma_1).
    """
    out = np.empty(z.shape + (max_degree + 1,))
    out[..., 0] = 1.0
    if max_degree >= 1:
        out[..., 1] = z
    for k in range(2, max_degree + 1):
        out[..., k] = (z * out[..., k - 1]
                       - np.sqrt(k - 1) * out[..., k - 2]) / np.sqrt(k)
    return out


def _compositions(K: int, s: int):
    """All compositions (a_1, ..., a_s) with a_i >= 1, sum <= K."""
    if s == 0:
        return [()]
    res = []
    for total in range(s, K + 1):
        for c in _compositions_exact(total, s):
            res.append(c)
    return res


def _compositions_exact(total: int, s: int):
    """All compositions of `total` into `s` parts of size >= 1."""
    if s == 1:
        return [(total,)]
    out = []
    for first in range(1, total - s + 2):
        for rest in _compositions_exact(total - first, s - 1):
            out.append((first,) + rest)
    return out


def enumerate_dictionary(d: int, K: int, q: int) -> list:
    """Multi-index set Λ_{K,q} = { α : 1 <= |α|_1 <= K, |supp(α)| <= q }.

    Excludes the zero multi-index. Each entry is a (supp_tuple, deg_tuple)
    pair where supp is sorted ascending and deg is the matching degrees.
    """
    out = []
    for s in range(1, q + 1):
        for supp in combinations(range(d), s):
            for deg in _compositions(K, s):
                out.append((tuple(supp), tuple(deg)))
    return out


def build_support_groups(A: list) -> list:
    """Group dictionary entries by their support pattern.

    Returns a list of (supp_tuple, deg_arr, global_idx_arr) where:
      supp_tuple: tuple of d-coords (length s in [1, q])
      deg_arr:    (n_g, s) int array of degree tuples in this group
      global_idx_arr: (n_g,) int array of indices into A

    This lets the per-axis Phi build use one fancy-indexed gather + product
    per group instead of one Python iteration per dictionary entry.
    """
    groups = {}
    for a, (supp, deg) in enumerate(A):
        groups.setdefault(supp, ([], []))
        groups[supp][0].append(deg)
        groups[supp][1].append(a)
    out = []
    for supp, (degs, idxs) in groups.items():
        deg_arr = np.array(degs, dtype=np.int64)
        idx_arr = np.array(idxs, dtype=np.int64)
        out.append((supp, deg_arr, idx_arr))
    return out


def _build_axis_meta(A: list, d: int):
    """For each axis i in [0, d), pre-compute:
       - local_idxs[i]: int array of dictionary indices touching axis i,
                        sorted ascending
       - groups_for_axis[i]: list of (supp, k_in_supp, deg_arr, local_pos)
         where local_pos[m] is the column position in the dense block for
         group entry m (consistent with the SORTED local_idxs[i]).

    Sorting local_idxs lets us safely accumulate into the upper triangle of
    the local Gram block via BLAS syrk and scatter into Amat under the
    convention "upper triangle = global rows < global cols".
    """
    groups_global = build_support_groups(A)

    raw_local = [[] for _ in range(d)]
    for (supp, deg_arr, idx_arr) in groups_global:
        for k, axis_i in enumerate(supp):
            for m, gidx in enumerate(idx_arr):
                raw_local[axis_i].append((int(gidx), supp, k, deg_arr, m))

    local_idxs_arr = []
    groups_for_axis = []
    for axis_i in range(d):
        items = raw_local[axis_i]
        items.sort(key=lambda t: t[0])
        gidxs = np.array([t[0] for t in items], dtype=np.int64)
        local_idxs_arr.append(gidxs)
        from collections import defaultdict
        per_supp = defaultdict(list)
        for local_pos, (gidx, supp, k, deg_arr, m) in enumerate(items):
            per_supp[(supp, k, id(deg_arr))].append(
                (local_pos, m, supp, k, deg_arr)
            )
        groups = []
        for key, entries in per_supp.items():
            local_pos_arr = np.array([e[0] for e in entries], dtype=np.int64)
            m_arr = np.array([e[1] for e in entries], dtype=np.int64)
            supp = entries[0][2]
            k = entries[0][3]
            deg_arr_full = entries[0][4]
            deg_arr_sub = deg_arr_full[m_arr]
            groups.append((supp, k, deg_arr_sub, local_pos_arr))
        groups_for_axis.append(groups)

    return local_idxs_arr, groups_for_axis


def _fill_phi_dense_for_axis(Phi_dense, Phi2_dense, H_chunk,
                             groups_for_axis_i):
    """Fill the dense per-axis Phi (and Phi^(2)) matrices for one axis.

    Phi_dense, Phi2_dense have shape (Nc, |S_i|) and are pre-zeroed by caller.
    For each multi-index alpha with axis i in support:
      Phi_dense[:, pos] = sqrt(alpha_i) * prod_{j in supp} H_{alpha_j - 1{j=i}}
      Phi2_dense[:, pos] = sqrt(alpha_i (alpha_i - 1)) * prod ... (second deriv)
    via the Hermite recurrence dH_alpha/dz_i = sqrt(alpha_i) H_{alpha - e_i}.
    """
    for (supp, k, deg_arr, local_pos) in groups_for_axis_i:
        s = len(supp)
        # First-derivative block
        deg_lookup = deg_arr.copy()
        deg_lookup[:, k] -= 1
        if s == 1:
            block = H_chunk[supp[0]][:, deg_lookup[:, 0]]
        elif s == 2:
            block = (H_chunk[supp[0]][:, deg_lookup[:, 0]]
                     * H_chunk[supp[1]][:, deg_lookup[:, 1]])
        elif s == 3:
            block = (H_chunk[supp[0]][:, deg_lookup[:, 0]]
                     * H_chunk[supp[1]][:, deg_lookup[:, 1]]
                     * H_chunk[supp[2]][:, deg_lookup[:, 2]])
        else:
            block = H_chunk[supp[0]][:, deg_lookup[:, 0]].copy()
            for l in range(1, s):
                block *= H_chunk[supp[l]][:, deg_lookup[:, l]]
        factor1 = np.sqrt(deg_arr[:, k].astype(np.float32))
        Phi_dense[:, local_pos] = factor1 * block

        # Second-derivative block (only for alpha_i >= 2)
        mask2 = deg_arr[:, k] >= 2
        if mask2.any():
            deg_lookup2 = deg_arr[mask2].copy()
            deg_lookup2[:, k] -= 2
            if s == 1:
                block2 = H_chunk[supp[0]][:, deg_lookup2[:, 0]]
            elif s == 2:
                block2 = (H_chunk[supp[0]][:, deg_lookup2[:, 0]]
                          * H_chunk[supp[1]][:, deg_lookup2[:, 1]])
            elif s == 3:
                block2 = (H_chunk[supp[0]][:, deg_lookup2[:, 0]]
                          * H_chunk[supp[1]][:, deg_lookup2[:, 1]]
                          * H_chunk[supp[2]][:, deg_lookup2[:, 2]])
            else:
                block2 = H_chunk[supp[0]][:, deg_lookup2[:, 0]].copy()
                for l in range(1, s):
                    block2 *= H_chunk[supp[l]][:, deg_lookup2[:, l]]
            a_k = deg_arr[mask2, k].astype(np.float32)
            factor2 = np.sqrt(a_k * (a_k - 1.0))
            Phi2_dense[:, local_pos[mask2]] = factor2 * block2


# =====================================================================
# Streaming (A, b) assembly and T-statistic computation
# =====================================================================

def build_A_b_streamed(Z_path: str, A: list, K: int, d: int, N: int,
                       N_chunk: int) -> tuple:
    """Streaming A, b assembly with per-axis dense blocks.

    For each axis i, we build only the |S_i| ~ |Lambda|·q/d columns that
    are nonzero, compute the small (|S_i|, |S_i|) Gram contribution via
    gemm, and scatter-add into the full (|Lambda|, |Lambda|) Gram.

    Args:
        Z_path:  npy memmap of (N, d) float32 rank-Gaussianized samples.
        A:       list of (supp, deg) multi-indices from enumerate_dictionary.
        K:       max total degree in A.
        d:       dimension.
        N:       number of samples to use from Z_path.
        N_chunk: row-chunk size for streaming.

    Returns:
        (Amat, bvec) of shape (|A|, |A|) and (|A|,), both float64.

    The (A, b) here are the Stage-1 score-matching normal equations:
        A_{αβ} = E_π_Z[∇H_α · ∇H_β]
        b_α    = E_π_Z[Σ_i (z_i ∂_i H_α - ∂_ii H_α)]
    """
    nA = len(A)
    Z = np.load(Z_path, mmap_mode='r')
    Amat = np.zeros((nA, nA), dtype=np.float64)
    bvec = np.zeros(nA, dtype=np.float64)

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for start in range(0, N, N_chunk):
        end = min(start + N_chunk, N)
        Z_chunk = np.asarray(Z[start:end])
        Nc = end - start
        H_chunk = np.empty((d, Nc, K + 1), dtype=np.float32)
        for j in range(d):
            H_chunk[j] = hermite_norm(Z_chunk[:, j], K).astype(np.float32)

        for axis_i in range(d):
            li = local_idxs[axis_i]
            n_local = len(li)
            if n_local == 0:
                continue
            Phi_dense = np.zeros((Nc, n_local), dtype=np.float32)
            Phi2_dense = np.zeros((Nc, n_local), dtype=np.float32)
            _fill_phi_dense_for_axis(Phi_dense, Phi2_dense, H_chunk,
                                     groups_for_axis[axis_i])

            Phi_dense_64 = Phi_dense.astype(np.float64)
            G_local = Phi_dense_64.T @ Phi_dense_64
            Amat[np.ix_(li, li)] += G_local

            bvec[li] += Z_chunk[:, axis_i].astype(np.float64) @ Phi_dense_64
            bvec[li] -= Phi2_dense.astype(np.float64).sum(axis=0)

            del Phi_dense, Phi2_dense, Phi_dense_64, G_local
        del H_chunk

    Amat /= N
    bvec /= N
    return Amat, bvec


def solve_ridge(Amat: np.ndarray, bvec: np.ndarray, A: list,
                lambda0: float) -> np.ndarray:
    """(A + lambda0 * diag(|alpha|^2)) theta = b via Cholesky."""
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=float)
    M = Amat + np.diag(lambda0 * total_deg ** 2)
    return np.linalg.solve(M, bvec)


def stream_T_at_basis(Z_path: str, theta: np.ndarray, A: list, K: int,
                      d: int, N: int, N_chunk: int,
                      T_out_path: Optional[str] = None) -> tuple:
    """Stream T = score residual at theta, accumulate T T^T / N and trC.

    If T_out_path given, also writes T (shape (d, N) float32) to a memmap.

    Args:
        Z_path:     rank-Gaussianized samples memmap.
        theta:      (|A|,) coefficients to evaluate the score at.
        A:          dictionary.
        K, d, N:    as build_A_b_streamed.
        N_chunk:    streaming chunk size.
        T_out_path: optional memmap to write the per-sample scores into.

    Returns:
        (TTdN, trC) where TTdN is (d, d) = T T^T / N and trC = trace(TTdN)
        = ||T||_F^2 / N.

    Per-axis dense build: for each axis i, only the |S_i| ~ |Lambda|·q/d
    nonzero columns of Phi_i are materialized. T_i = Phi_i_dense @ theta[S_i].
    """
    Z = np.load(Z_path, mmap_mode='r')
    TTdN = np.zeros((d, d), dtype=np.float64)
    sumsq = 0.0
    T_mm = None
    if T_out_path is not None:
        T_mm = np.lib.format.open_memmap(
            T_out_path, mode='w+', dtype=np.float32, shape=(d, N)
        )

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for start in range(0, N, N_chunk):
        end = min(start + N_chunk, N)
        Z_chunk = np.asarray(Z[start:end])
        Nc = end - start
        H_chunk = np.empty((d, Nc, K + 1), dtype=np.float32)
        for j in range(d):
            H_chunk[j] = hermite_norm(Z_chunk[:, j], K).astype(np.float32)

        T_chunk = np.zeros((d, Nc), dtype=np.float64)
        for axis_i in range(d):
            li = local_idxs[axis_i]
            n_local = len(li)
            if n_local == 0:
                continue
            Phi_dense = np.zeros((Nc, n_local), dtype=np.float32)
            Phi2_dense_unused = np.zeros((Nc, n_local), dtype=np.float32)
            _fill_phi_dense_for_axis(
                Phi_dense, Phi2_dense_unused, H_chunk,
                groups_for_axis[axis_i],
            )
            del Phi2_dense_unused
            theta_local = theta[li].astype(np.float64)
            T_chunk[axis_i, :] = Phi_dense.astype(np.float64) @ theta_local
            del Phi_dense

        TTdN += T_chunk @ T_chunk.T
        sumsq += float(np.sum(T_chunk * T_chunk))
        if T_mm is not None:
            T_mm[:, start:end] = T_chunk.astype(np.float32)
        del T_chunk, H_chunk
    TTdN /= N
    trC = sumsq / N
    if T_mm is not None:
        T_mm.flush()
    return TTdN, trC


def top_r_eigs(M: np.ndarray, r: int) -> tuple:
    """Eigendecomp of d x d symmetric M -> (V_r, top eigvals, all eigvals).

    Returns eigenvalues in descending order; V_r is the corresponding
    matrix of leading eigenvectors.
    """
    w, V = np.linalg.eigh(M)
    order = np.argsort(-w)
    w = w[order]
    V = V[:, order]
    return V[:, :r], w[:r], w


# =====================================================================
# Marginal CDFs (empirical, memmap-backed)
# =====================================================================

class MarginalCDFs:
    """Empirical marginal CDFs from a (large) pool, backed by per-coord
    sorted arrays kept on disk as a memmap.

    Strict positivity: returned u-values are clipped to (1e-12, 1 - 1e-12)
    so Φ^{-1} stays finite.
    """

    def __init__(self, sorted_pool_path: str, N_pool: int, d: int):
        self.path = sorted_pool_path
        self.N_pool = N_pool
        self.d = d
        self.sorted_pool = np.load(sorted_pool_path, mmap_mode='r')

    @classmethod
    def build(cls, eta_pool: np.memmap, sorted_path: str) -> 'MarginalCDFs':
        """Construct per-coord sorted memmap from a large eta pool.

        Writes to a `.tmp` path and atomically renames on completion, with
        per-column flushes so that a kill mid-loop leaves a partial file at
        `.tmp` that is detectably incomplete (the real path is created only
        on full success). The atomic rename ensures the final file at
        `sorted_path` is either complete or absent.
        """
        N, d = eta_pool.shape
        tmp_path = sorted_path + '.tmp'
        # Defensively remove any prior tmp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out = np.lib.format.open_memmap(
            tmp_path, mode='w+', dtype=np.float32, shape=(N, d),
        )
        try:
            for j in range(d):
                col = np.array(eta_pool[:, j], dtype=np.float32, copy=True)
                col.sort()
                out[:, j] = col
                out.flush()
            del out
            os.replace(tmp_path, sorted_path)
        except BaseException:
            try:
                del out
            except Exception:
                pass
            # leave tmp on disk for inspection but don't promote
            raise
        return cls(sorted_path, N, d)

    def cdf(self, eta: np.ndarray) -> np.ndarray:
        """F_i(eta_i) componentwise, shape (N, d) -> (N, d) in (0, 1)."""
        N, d = eta.shape
        out = np.empty_like(eta, dtype=np.float64)
        for i in range(d):
            ranks = np.searchsorted(
                self.sorted_pool[:, i], eta[:, i], side='right'
            )
            out[:, i] = (ranks + 0.5) / (self.N_pool + 1)
        return np.clip(out, 1e-12, 1 - 1e-12)


def _validate_sorted_pool(sorted_path: str, N_pool: int, d: int,
                           n_probe: int = 40) -> tuple:
    """Check that a sorted_pool memmap is well-formed.

    A valid pool has, for each column j: values ascending, with non-trivial
    spread, and the upper-half should contain non-zero values for typical
    noise distributions (eta_pool has zero mean and substantial variance,
    so its max should be well above zero).

    We probe `n_probe` evenly-spaced indices per column rather than reading
    the whole array, because 20M-row memmap is 80 MB per column.

    Returns (ok: bool, reason: str). reason is empty if ok.
    """
    if not os.path.exists(sorted_path):
        return False, f"file missing: {sorted_path}"
    try:
        arr = np.load(sorted_path, mmap_mode='r')
    except Exception as e:
        return False, f"could not load: {e}"
    if arr.shape != (N_pool, d):
        return False, f"shape mismatch: got {arr.shape}, expected ({N_pool}, {d})"

    # Probe per column
    probe_idxs = np.linspace(0, N_pool - 1, n_probe).astype(np.int64)
    for j in range(d):
        samples = np.asarray(arr[probe_idxs, j], dtype=np.float64)
        # Strictly non-decreasing
        if not np.all(samples[1:] >= samples[:-1] - 1e-6):
            return False, (f"col {j} not sorted at probe indices "
                            f"(samples: {samples[:5]} ... {samples[-5:]})")
        # Non-trivial spread
        if samples[-1] - samples[0] < 1e-6:
            return False, (f"col {j} has trivial spread "
                            f"(min={samples[0]:.6g}, max={samples[-1]:.6g})")
        # Upper half should have nonzero positive values for noise marginals
        upper_half = samples[n_probe // 2:]
        if not np.any(upper_half > 1e-6):
            return False, (f"col {j} upper half is all zeros "
                            f"(suggests partial write)")
    return True, ""


# =====================================================================
# Memmap helpers
# =====================================================================

def sample_to_memmap(
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    N_total: int, chunk: int, seed: int, d_obs: int, path: str,
) -> np.memmap:
    """Sample N_total samples from noise_sampler into a memmap on disk.

    The sampler is called once per chunk with a single RNG; this is
    deterministic given (seed, chunk schedule).
    """
    rng = np.random.default_rng(seed)
    mm = np.lib.format.open_memmap(
        path, mode='w+', dtype=np.float32, shape=(N_total, d_obs),
    )
    for start in range(0, N_total, chunk):
        end = min(start + chunk, N_total)
        mm[start:end] = noise_sampler(end - start, rng).astype(np.float32)
    mm.flush()
    return mm


def rank_gaussianize_to_memmap(eta_path: str, marginals: MarginalCDFs,
                                out_path: str, chunk: int) -> np.memmap:
    """Rank-Gaussianize eta -> Z = Phi^{-1}(F(eta)) into a memmap."""
    eta = np.load(eta_path, mmap_mode='r')
    N, d = eta.shape
    Z = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32, shape=(N, d),
    )
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        u = marginals.cdf(np.asarray(eta[start:end]))
        Z[start:end] = stats.norm.ppf(u).astype(np.float32)
    Z.flush()
    return Z


def materialize_Z(
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    d_obs: int, marginals: MarginalCDFs, N: int, seed: int,
    scratch_dir: str, kind: str, N_chunk: int,
) -> str:
    """Sample N from noise_sampler, rank-Gaussianize via marginals, store Z.

    Returns the path to the Z memmap (.npy). Idempotent: if Z exists, no-op.

    `kind` is a tag like "ref", "sat", "emp" used in the cache filenames so
    that simultaneous stages don't collide.
    """
    eta_path = os.path.join(scratch_dir, f'eta_{kind}_N{N}_seed{seed}.npy')
    Z_path = os.path.join(scratch_dir, f'Z_{kind}_N{N}_seed{seed}.npy')
    if os.path.exists(Z_path):
        return Z_path
    if not os.path.exists(eta_path):
        sample_to_memmap(noise_sampler, N, N_chunk, seed, d_obs, eta_path)
    rank_gaussianize_to_memmap(eta_path, marginals, Z_path, N_chunk)
    return Z_path


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    """Stage-pipeline configuration. The caller passes d_obs and the noise
    sampler to the stage functions."""
    # Observation dimension (benchmark-dependent; caller must set)
    d_obs: int = 20

    # Hermite-basis truncation (Stage 1 outer)
    K_outer: int = 4
    q_outer: int = 2
    r: int = 4

    # Sample-size scan
    N_scan: tuple = (500, 2500, 12500, 50000, 200_000)
    n_seeds: int = 8

    # Reference computations
    N_pop: int = 20_000_000        # pool for marginal CDFs
    N_ref: int = 1_000_000          # reference for population C_Lambda
    N_sat: int = 1_000_000          # saturation curve (fixed N)

    # Saturation ladder
    sat_bases: tuple = ((2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
                        (4, 3), (5, 3), (6, 3))

    # Regularization
    # Outer Hermite-score Tikhonov scale: lambda = lambda0_factor_emp * ||A||_op.
    # Must stay in sync with cas.hermite_score_matching.LAMBDA0_FACTOR_DEFAULT.
    lambda0_factor_emp: float = 1e-4
    sat_lambda_mode: str = 'production'    # 'production' | 'fixed'
    sat_lambda_abs: float = 1e-6
    ref_lambda_mode: str = 'production'
    ref_lambda_abs: float = 1e-6

    # Streaming chunk size. Per-axis Phi at chunk size N_chunk is
    # N_chunk * |Lambda| * 4 bytes (float32). Two of those (Phi + Phi^2)
    # live at the same time, plus the Gram matrix |Lambda|^2 * 8 bytes
    # (float64). Larger chunks amortize BLAS setup cost but raise memory
    # peak.
    N_chunk: int = 80_000

    # Compute
    n_workers: int = 12

    # Memory budget for Stage 2. Bases whose estimated peak exceeds this
    # are skipped with a warning. Set well below your physical RAM.
    sat_mem_budget_gb: float = 35.0

    # Seeds
    seed_pop: int = 99999
    seed_ref: int = 99998
    seed_sat: int = 99997
    seeds_emp_offset: int = 7000

    def quick(self) -> "Config":
        """Fast smoke test config (~2 min)."""
        return Config(
            d_obs=self.d_obs,
            K_outer=4, q_outer=2, r=self.r,
            N_scan=(500, 2500),
            n_seeds=2,
            N_pop=50_000, N_ref=10_000, N_sat=10_000,
            sat_bases=((2, 2), (3, 2), (4, 2)),
            lambda0_factor_emp=self.lambda0_factor_emp,
            sat_lambda_mode=self.sat_lambda_mode,
            sat_lambda_abs=self.sat_lambda_abs,
            ref_lambda_mode=self.ref_lambda_mode,
            ref_lambda_abs=self.ref_lambda_abs,
            N_chunk=2000,
            n_workers=min(2, self.n_workers),
            sat_mem_budget_gb=4.0,
        )


# =====================================================================
# Stages
# =====================================================================

def stage_pool(
    cfg: Config,
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    scratch_dir: str,
) -> MarginalCDFs:
    """Stage 1: build the eta pool and per-coord sorted memmap.

    Outputs into scratch_dir:
      pool_eta_N{N_pop}.npy
      pool_sorted_N{N_pop}.npy

    Returns a MarginalCDFs wrapper around the sorted memmap.

    Idempotent: re-running with the same N_pop reuses existing artifacts.
    """
    os.makedirs(scratch_dir, exist_ok=True)
    eta_path = os.path.join(scratch_dir, f'pool_eta_N{cfg.N_pop}.npy')
    sorted_path = os.path.join(scratch_dir, f'pool_sorted_N{cfg.N_pop}.npy')

    if not os.path.exists(eta_path):
        print(f"[Stage 1] sampling eta pool: N_pop={cfg.N_pop}", flush=True)
        t = time.time()
        sample_to_memmap(noise_sampler, cfg.N_pop, cfg.N_chunk, cfg.seed_pop,
                         cfg.d_obs, eta_path)
        print(f"   eta saved in {time.time()-t:.0f}s", flush=True)

    # Validate sorted pool; rebuild if corrupt or missing.
    need_rebuild = False
    if not os.path.exists(sorted_path):
        need_rebuild = True
        print(f"[Stage 1] sorted pool missing -- building", flush=True)
    else:
        ok, reason = _validate_sorted_pool(
            sorted_path, cfg.N_pop, cfg.d_obs,
        )
        if not ok:
            need_rebuild = True
            print(f"[Stage 1] sorted pool INVALID: {reason}", flush=True)
            print(f"[Stage 1]   removing corrupt {sorted_path} and rebuilding",
                  flush=True)
            os.remove(sorted_path)
        else:
            print(f"[Stage 1] sorted pool exists and validates -- reusing",
                  flush=True)

    if need_rebuild:
        t = time.time()
        eta_mm = np.load(eta_path, mmap_mode='r')
        MarginalCDFs.build(eta_mm, sorted_path)
        # Verify after build
        ok, reason = _validate_sorted_pool(
            sorted_path, cfg.N_pop, cfg.d_obs,
        )
        if not ok:
            raise RuntimeError(
                f"sorted pool failed validation immediately after build: "
                f"{reason}"
            )
        print(f"   sorted pool saved and validated in {time.time()-t:.0f}s",
              flush=True)
    return MarginalCDFs(sorted_path, cfg.N_pop, cfg.d_obs)


def _sat_one_basis(
    cfg: Config, K: int, q: int, Z_path: str, blas_threads: int,
) -> dict:
    """Compute tr(C_Lambda) at one (K, q) basis."""
    A = enumerate_dictionary(cfg.d_obs, K, q)
    t = time.time()

    def _do():
        Amat, bvec = build_A_b_streamed(
            Z_path, A, K, cfg.d_obs, cfg.N_sat, cfg.N_chunk
        )
        if cfg.sat_lambda_mode == 'production':
            lam = cfg.lambda0_factor_emp * np.linalg.norm(Amat, ord=2)
        else:
            lam = cfg.sat_lambda_abs
        th = solve_ridge(Amat, bvec, A, lam)
        _, trC_K = stream_T_at_basis(
            Z_path, theta=th, A=A, K=K, d=cfg.d_obs, N=cfg.N_sat,
            N_chunk=cfg.N_chunk,
        )
        return lam, trC_K

    if blas_threads and blas_threads > 0:
        with threadpool_limits(limits=int(blas_threads)):
            lambda0, trC_K = _do()
    else:
        lambda0, trC_K = _do()

    return {
        'K': K, 'q': q,
        'dict_size': len(A),
        'trC': float(trC_K),
        'lambda0': float(lambda0),
        'blas_threads': int(blas_threads) if blas_threads else 0,
        'time_s': time.time() - t,
    }


def stage_saturation(
    cfg: Config,
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    marginals: MarginalCDFs,
    scratch_dir: str,
    cache_path: str,
) -> dict:
    """Stage 2: tr(C_Lambda) saturation curve across the (K, q) ladder.

    Caches per-basis results into cache_path (atomic JSON, append-only by
    (K, q) key so interrupted runs resume).

    Skips bases that would exceed sat_mem_budget_gb with a warning.

    Returns a dict with keys:
        rows                 : list of per-basis dicts
        N_sat                : cfg.N_sat
        trC_estimate_full    : max trC over completed bases (≈ tr C)
        trC_at_deployment    : trC at (K_outer, q_outer)
        basis_truncation_sq_estimate : T2² estimate
    """
    rows = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
        rows = existing.get('rows', [])
        if rows:
            done_keys = {(r['K'], r['q']) for r in rows}
            print(f"[Stage 2] resuming saturation "
                  f"({cfg.sat_lambda_mode}): {len(rows)} bases cached: "
                  f"{sorted(done_keys)}", flush=True)
        all_keys = set(tuple(b) for b in cfg.sat_bases)
        if rows and all_keys.issubset({(r['K'], r['q']) for r in rows}):
            print(f"[Stage 2] all bases cached, returning", flush=True)
            return existing

    print(f"[Stage 2] tr(C_Lambda) saturation, lambda_mode="
          f"{cfg.sat_lambda_mode}", flush=True)
    Z_path = materialize_Z(
        noise_sampler, cfg.d_obs, marginals, cfg.N_sat, cfg.seed_sat,
        scratch_dir, 'sat', cfg.N_chunk,
    )

    done_keys = {(r['K'], r['q']) for r in rows}

    def _save_partial():
        rows.sort(key=lambda r: (r['K'], r['q']))
        out = {
            'rows': rows, 'N_sat': cfg.N_sat,
            'lambda_mode': cfg.sat_lambda_mode,
            'sat_lambda_abs': cfg.sat_lambda_abs,
            'lambda0_factor_emp': cfg.lambda0_factor_emp,
            'sat_mem_budget_gb': cfg.sat_mem_budget_gb,
            'partial': True,
        }
        if rows:
            out['trC_estimate_full'] = max(r['trC'] for r in rows)
            depl_match = [
                r['trC'] for r in rows
                if r['K'] == cfg.K_outer and r['q'] == cfg.q_outer
            ]
            if depl_match:
                out['trC_at_deployment'] = depl_match[0]
                out['basis_truncation_sq_estimate'] = max(
                    out['trC_estimate_full'] - depl_match[0], 0.0
                )
        tmp = cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, cache_path)
        return out

    for (K, q) in cfg.sat_bases:
        if (K, q) in done_keys:
            print(f"   skipping (K={K}, q={q}): already cached", flush=True)
            continue
        peak = estimate_peak_gb(K, q, cfg.d_obs, cfg.N_chunk)
        if peak > cfg.sat_mem_budget_gb:
            print(f"   skipping (K={K}, q={q}): peak {peak:.1f} GB exceeds "
                  f"sat_mem_budget_gb={cfg.sat_mem_budget_gb:.1f} GB",
                  flush=True)
            continue
        print(f"   starting (K={K}, q={q}) [peak ~ {peak:.2f} GB]",
              flush=True)
        r = _sat_one_basis(cfg, K, q, Z_path, blas_threads=0)
        print(f"   done (K={K}, q={q}): tr(C)={r['trC']:.4f}, "
              f"lambda0={r['lambda0']:.4g} ({r['time_s']:.0f}s)",
              flush=True)
        rows.append(r)
        done_keys.add((K, q))
        _save_partial()
        print(f"   [saved partial — safe to interrupt]", flush=True)

    # Final save with partial=False
    rows.sort(key=lambda r: (r['K'], r['q']))
    out = {
        'rows': rows, 'N_sat': cfg.N_sat,
        'lambda_mode': cfg.sat_lambda_mode,
        'sat_lambda_abs': cfg.sat_lambda_abs,
        'lambda0_factor_emp': cfg.lambda0_factor_emp,
        'sat_mem_budget_gb': cfg.sat_mem_budget_gb,
    }
    if rows:
        trC_full = max(r['trC'] for r in rows)
        trC_depl_match = [
            r['trC'] for r in rows
            if r['K'] == cfg.K_outer and r['q'] == cfg.q_outer
        ]
        trC_depl = trC_depl_match[0] if trC_depl_match else float('nan')
        out['trC_estimate_full'] = trC_full
        out['trC_at_deployment'] = trC_depl
        out['basis_truncation_sq_estimate'] = (
            max(trC_full - trC_depl, 0.0) if trC_depl_match else float('nan')
        )
        print(f"   tr(C) richest={trC_full:.4f}, deployment={trC_depl:.4f}, "
              f"basis-trunc^2={out['basis_truncation_sq_estimate']:.4f}",
              flush=True)
    tmp = cache_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, cache_path)
    return out


def stage_reference(
    cfg: Config,
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    marginals: MarginalCDFs,
    scratch_dir: str,
    cache_path: str,
    T_ref_path: Optional[str] = None,
) -> dict:
    """Stage 3: reference C_Lambda at the deployment basis.

    Produces:
      - trC_Lambda, top-r eigvals, V_r (the reference eigenbasis)
      - dict_size, lambda0
      - T_ref memmap on disk (used by Stage 4 for ||V̂^T T_ref||² computation)

    Idempotent: rerun returns the cached result.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            out = json.load(f)
        if 'trC' in out and 'V_r' in out:
            print(f"[Stage 3] cached reference: trC_Lambda={out['trC']:.4f}, "
                  f"E_r={out['trC'] - sum(out['eigvals_top']):.4f}",
                  flush=True)
            return out

    print(f"[Stage 3] reference C_Lambda at N_ref={cfg.N_ref}, "
          f"(K,q)=({cfg.K_outer},{cfg.q_outer})", flush=True)
    Z_path = materialize_Z(
        noise_sampler, cfg.d_obs, marginals, cfg.N_ref, cfg.seed_ref,
        scratch_dir, 'ref', cfg.N_chunk,
    )
    A = enumerate_dictionary(cfg.d_obs, cfg.K_outer, cfg.q_outer)
    print(f"   |Lambda|={len(A)}", flush=True)
    t = time.time()
    Amat, bvec = build_A_b_streamed(
        Z_path, A, cfg.K_outer, cfg.d_obs, cfg.N_ref, cfg.N_chunk,
    )
    print(f"   A,b in {time.time()-t:.0f}s", flush=True)
    if cfg.ref_lambda_mode == 'production':
        lambda0 = cfg.lambda0_factor_emp * np.linalg.norm(Amat, ord=2)
    else:
        lambda0 = cfg.ref_lambda_abs
    theta = solve_ridge(Amat, bvec, A, lambda0)
    t = time.time()
    if T_ref_path is None:
        T_ref_path = os.path.join(
            scratch_dir,
            f'ref_T_K{cfg.K_outer}_q{cfg.q_outer}_N{cfg.N_ref}.npy',
        )
    TTdN, trC = stream_T_at_basis(
        Z_path, theta=theta, A=A, K=cfg.K_outer, d=cfg.d_obs, N=cfg.N_ref,
        N_chunk=cfg.N_chunk, T_out_path=T_ref_path,
    )
    print(f"   T-stream + save in {time.time()-t:.0f}s", flush=True)
    V_r, eigvals_top, _ = top_r_eigs(TTdN, cfg.r)
    print(f"   trC_Lambda={trC:.4f}, top eigvals={eigvals_top}, "
          f"lambda0={lambda0:.4g}", flush=True)
    out = {
        'trC': float(trC),
        'eigvals_top': [float(x) for x in eigvals_top],
        'V_r': V_r.tolist(),
        'dict_size': len(A),
        'lambda0': float(lambda0),
        'K': cfg.K_outer, 'q': cfg.q_outer, 'N_ref': cfg.N_ref,
        'T_path': T_ref_path,
    }
    tmp = cache_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, cache_path)
    return out


def _emp_one_cell(
    cfg_dict: dict,
    noise_sampler_pickle: bytes,
    N: int, seed: int,
    V_r_ref_list: list,
    trC_Lambda_ref: float,
    T_ref_path: str,
    scratch_dir: str,
    sorted_pool_path: str,
    sorted_pool_N: int,
) -> dict:
    """One (N, seed) cell of the Stage 4 scan.

    Runs in a worker process; uses threadpoolctl to pin BLAS to 1 thread.
    Passes config and noise_sampler through as serializable forms.
    """
    cfg = Config(**cfg_dict)
    import pickle
    noise_sampler = pickle.loads(noise_sampler_pickle)

    with threadpool_limits(limits=1):
        marginals = MarginalCDFs(sorted_pool_path, sorted_pool_N, cfg.d_obs)
        Z_path = materialize_Z(
            noise_sampler, cfg.d_obs, marginals, N,
            cfg.seeds_emp_offset + seed, scratch_dir, 'emp', cfg.N_chunk,
        )

        A = enumerate_dictionary(cfg.d_obs, cfg.K_outer, cfg.q_outer)
        Amat, bvec = build_A_b_streamed(
            Z_path, A, cfg.K_outer, cfg.d_obs, N, cfg.N_chunk,
        )
        lambda0 = cfg.lambda0_factor_emp * np.linalg.norm(Amat, ord=2)
        theta = solve_ridge(Amat, bvec, A, lambda0)
        TTdN, trC_emp = stream_T_at_basis(
            Z_path, theta=theta, A=A, K=cfg.K_outer, d=cfg.d_obs, N=N,
            N_chunk=cfg.N_chunk,
        )
        V_hat, eigvals_emp, _ = top_r_eigs(TTdN, cfg.r)

        V_r_ref = np.array(V_r_ref_list)
        s = np.linalg.svd(V_hat.T @ V_r_ref, compute_uv=False)
        s = np.clip(s, 0, 1)
        sin_theta = float(np.sqrt(np.sum(1 - s ** 2)))

        # Stage-1 oracle KL (LSI upper bound, not the true KL)
        T_ref = np.load(T_ref_path, mmap_mode='r')
        N_ref = T_ref.shape[1]
        captured_sq = 0.0
        block_sz = max(cfg.N_chunk, 50000)
        for s2 in range(0, N_ref, block_sz):
            e = min(s2 + block_sz, N_ref)
            block = np.asarray(T_ref[:, s2:e], dtype=np.float64)
            proj = V_hat.T @ block
            captured_sq += float(np.sum(proj * proj))
        captured_sq /= N_ref
        oracle_kl = 0.5 * (trC_Lambda_ref - captured_sq)

    return {
        'N': N, 'seed': seed,
        'sin_theta_F': sin_theta,
        'oracle_KL_bound': float(oracle_kl),
        'captured_sq': float(captured_sq),
        'trC_emp': float(trC_emp),
        'eigvals_emp_top': [float(x) for x in eigvals_emp],
        # V_hat is needed by exp_stage1_reference for the
        # conditional Monte Carlo of the true Stage-1 oracle KL. d x r = 80
        # floats per cell on banana; trivial overhead.
        'V_hat': V_hat.tolist(),
    }


def stage_scan(
    cfg: Config,
    noise_sampler: Callable[[int, np.random.Generator], np.ndarray],
    ref: dict,
    marginals: MarginalCDFs,
    scratch_dir: str,
    cache_path: str,
    n_workers: Optional[int] = None,
) -> list:
    """Stage 4: per-(N, seed) scan.

    For each cell, fits Vhat_r at the deployment basis on N samples, and
    computes ||sin Theta(Vhat_r, V_r_ref)||_F and the Stage-1 LSI oracle KL
    upper bound 0.5 (trC_Lambda - ||V̂^T T_ref||²/N_ref).

    Resumable: cells already in cache_path are skipped.

    Returns the merged list of cell-dicts.
    """
    n_workers = n_workers or cfg.n_workers
    existing = []
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
    done = {(r['N'], r['seed']) for r in existing}
    todo = [
        (N, s) for N in cfg.N_scan for s in range(cfg.n_seeds)
        if (N, s) not in done
    ]
    if not todo:
        print(f"[Stage 4] all {len(existing)} (N, seed) cells cached",
              flush=True)
        return existing
    print(f"[Stage 4] {len(todo)} cells, {n_workers} workers", flush=True)

    cfg_dict = asdict(cfg)
    V_r_ref_list = ref['V_r']
    trC_Lambda_ref = float(ref['trC'])
    T_ref_path = ref['T_path']

    # Pickle the noise_sampler once for worker dispatch
    import pickle
    noise_sampler_pickle = pickle.dumps(noise_sampler)

    args_list = [
        (cfg_dict, noise_sampler_pickle, N, s, V_r_ref_list, trC_Lambda_ref,
         T_ref_path, scratch_dir, marginals.path, marginals.N_pool)
        for (N, s) in todo
    ]

    def _save(items):
        tmp = cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, cache_path)

    if n_workers <= 1:
        for args in args_list:
            t = time.time()
            res = _emp_one_cell(*args)
            res['time_s'] = time.time() - t
            existing.append(res)
            _save(existing)
            print(f"  N={res['N']}, seed={res['seed']}: "
                  f"sin Theta={res['sin_theta_F']:.4f}, "
                  f"oracle KL={res['oracle_KL_bound']:.4f} "
                  f"({res['time_s']:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_emp_one_cell, *args): args for args in args_list
            }
            for fut in as_completed(futures):
                res = fut.result()
                existing.append(res)
                _save(existing)
                print(f"  N={res['N']}, seed={res['seed']}: "
                      f"sin Theta={res['sin_theta_F']:.4f}, "
                      f"oracle KL={res['oracle_KL_bound']:.4f}", flush=True)
    return existing
