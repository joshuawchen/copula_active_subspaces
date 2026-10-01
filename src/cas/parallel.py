"""Spawn-safe multiprocessing with BLAS pinned to one thread per worker:
run_parallel(fn, tasks, n_workers) and run_parallel_init(fn, tasks,
n_workers, initializer, initargs), both serial when n_workers <= 1.
"""
from __future__ import annotations

from .utils import resolve_workers, run_parallel, run_parallel_init

__all__ = ["resolve_workers", "run_parallel", "run_parallel_init"]
