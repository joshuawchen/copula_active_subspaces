"""Compares the quadrature estimator of cas.quad_marginalization with Laplace
importance sampling on Examples 1-3 at a basis near the reference: agreement
within the importance-sampling error, and run time.

Run: PYTHONPATH=src python3 scripts/validate_quad_vs_is.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from cas.oracle_kl_marginalization import sample_pi_U
from cas.conditional_kl import marginalize_pi_U_with_inner_kl
from cas.noise import L_CHOL as _MIX_BAN
from cas.even_fold import L_CHOL_EF as _MIX_EF
from cas.conformal_cube import L_CHOL_C3 as _MIX_C3

_MIX = {"banana": _MIX_BAN, "even_fold": _MIX_EF, "conformal_cube": _MIX_C3}
N_U = 12
R = 4
TILT = 0.2          # sinTheta ~ 0.2 near-active basis, like a mid/large-N V_hat


def near_active_Vr(mixing, seed=0):
    d = np.asarray(mixing).shape[0]
    Va = np.linalg.qr(np.asarray(mixing)[:, :R])[0]        # exact active subspace
    rng = np.random.default_rng(seed)
    return np.linalg.qr(Va + TILT * rng.standard_normal((d, R)))[0]


print(f"quad vs Laplace-IS at a near-active subspace (sinTheta~{TILT}, r={R})\n")
for bench in ["banana", "even_fold", "conformal_cube"]:
    V_r = near_active_Vr(_MIX[bench], seed=0)
    sinT = float(np.linalg.norm(
        np.linalg.svd(V_r.T @ np.linalg.qr(np.asarray(_MIX[bench])[:, :R])[0],
                      compute_uv=False).clip(-1, 1)))
    U = sample_pi_U(N_U, V_r, bench, seed=1)

    t0 = time.time()
    is_res = marginalize_pi_U_with_inner_kl(
        U, V_r, bench, M_aux=100_000, n_mode_starts=20, seed=7,
        inner_sampler="laplace_is",
    )
    t_is = time.time() - t0

    t0 = time.time()
    q_res = marginalize_pi_U_with_inner_kl(
        U, V_r, bench, seed=7, inner_sampler="quad",
    )
    t_q = time.time() - t0

    dk = q_res["inner_kl"] - is_res["inner_kl"]
    dl = q_res["log_pi_U"] - is_res["log_pi_U"]
    n_skip = int(np.sum(q_res["inner_sampler_used"] != "quad"))
    print(f"=== {bench}  (N_u={N_U}, quad cells={N_U - n_skip}, fallbacks={n_skip}) ===")
    print(f"  inner_kl : mean|d|={np.mean(np.abs(dk)):.4f}  max|d|={np.max(np.abs(dk)):.4f}"
          f"   IS inner-KL SE={np.mean(is_res['inner_kl_se']):.4f}")
    print(f"  log_pi_U : mean|d|={np.mean(np.abs(dl)):.4f}  max|d|={np.max(np.abs(dl)):.4f}")
    print(f"  KL means : quad={np.mean(q_res['inner_kl']):.4f}   IS={np.mean(is_res['inner_kl']):.4f}")
    print(f"  TIME     : IS={t_is:.1f}s   quad={t_q:.1f}s   speedup={t_is / max(t_q, 1e-9):.1f}x\n")
print(f"PASS if mean|d| ~<= IS inner-KL SE on all three, and quad is faster.")
