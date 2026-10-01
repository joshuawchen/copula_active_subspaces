"""Monte Carlo estimate of E[log pi_n(eta)] for Example 1 from 10^6 samples,
used in the noise KL divergence. Writes cache/v81_noise_entropy.npz.
"""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from cas.config import CACHE_DIR
from cas import sample_noise, log_noise_density


def main():
    N_samples = 1_000_000
    seed = 99999
    rng = np.random.default_rng(seed)
    print(f"Sampling {N_samples} banana noise samples and computing log π_η...")
    import time
    t0 = time.time()
    eta = sample_noise(N_samples, rng)
    log_p = log_noise_density(eta)
    neg_H = float(log_p.mean())
    sem = float(log_p.std(ddof=1) / np.sqrt(N_samples))
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")
    print(f"  neg_H = E[log π_η] = {neg_H:.6f}")
    print(f"  SEM = {sem:.6f}")

    out_path = os.path.join(CACHE_DIR, "v81_noise_entropy.npz")
    np.savez(out_path, neg_H=neg_H, n_samples=N_samples, sem=sem)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
