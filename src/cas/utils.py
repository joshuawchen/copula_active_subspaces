"""Shared helpers: BLAS thread pinning (import before numpy), process pools
and cache I/O.
"""
from __future__ import annotations
import os
import sys
import time

# Pin BLAS to 1 thread per process. Required before numpy import.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np                                                  # noqa: E402
from multiprocessing import Pool, cpu_count                         # noqa: E402

from .config import N_WORKERS                                        # noqa: E402


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------
def resolve_workers(n_tasks, override=None):
    """Number of worker processes for ``n_tasks`` tasks.

    Returns 1 (serial) when there's only one task or workers were forced to 1.
    """
    n = override if override is not None else N_WORKERS
    if n is None:
        n = max(1, cpu_count() - 1)
    return max(1, min(int(n), n_tasks))


def run_parallel(fn, tasks, n_workers, desc=''):
    """Map ``fn`` across ``tasks`` with multiprocessing.

    ``fn`` must be importable at module scope (so that workers can pickle it).
    Returns a list of results in input order.
    """
    if n_workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(fn, tasks)
    if desc:
        print(f'  [{desc}] {len(tasks)} tasks on {n_workers} workers '
              f'in {time.time() - t0:.1f}s', flush=True)
    return results


def run_parallel_init(fn, tasks, n_workers, initializer, initargs, desc=''):
    """Like run_parallel, but each worker is initialized with `initializer(*initargs)`.

    Used to share read-only fitted-model state with workers under spawn-mode
    multiprocessing (macOS default), where module globals don't propagate.
    """
    if n_workers <= 1 or len(tasks) <= 1:
        # Run sequentially in main process; mimic initializer for consistency.
        initializer(*initargs)
        return [fn(t) for t in tasks]
    t0 = time.time()
    with Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        results = pool.map(fn, tasks)
    if desc:
        print(f'  [{desc}] {len(tasks)} tasks on {n_workers} workers '
              f'in {time.time() - t0:.1f}s', flush=True)
    return results


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def save_cache(path, **kwargs):
    """Save ``kwargs`` as a single .npz at ``path``. Atomic via temp file."""
    tmp = path + '.tmp'
    # Open as binary file ourselves -- np.savez_compressed adds '.npz' to
    # string paths that don't already end in '.npz'.
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **kwargs)
    os.replace(tmp, path)


def load_cache(path):
    """Load .npz dict-of-arrays from ``path``, or None if missing."""
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def fmt_dur(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m'
    if m:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def banner(title):
    """Print a section header."""
    bar = '=' * 72
    print(f'\n{bar}\n  {title}\n{bar}', flush=True)
