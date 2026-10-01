"""Part II supplement: subspace-recovery rate of CAS on a Gaussian copula with
closed-form C, against the O(N^{-1/2}) reference. Writes rate_plot.png.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make cas.py importable regardless of where we run from
HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, 'src'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas import cas_fit, sin_theta_F  # noqa: E402

# Worker globals (set by initializer for parallel execution).
_WORKER_STATE = {}


def _init_worker(d, r, rho, K, q, seed0, R_serialized, V_oracle_r_serialized):
    """Initialize a worker process with the read-only experiment state."""
    _WORKER_STATE.update(
        d=d, r=r, rho=rho, K=K, q=q, seed0=seed0,
        R=np.array(R_serialized),
        V_oracle_r=np.array(V_oracle_r_serialized),
    )


def _run_one_rep(task):
    """Single CAS fit for one (i_N, N, rep, n_rep) tuple. Reads worker state."""
    i_N, N, rep, n_rep = task
    d = _WORKER_STATE['d']
    r = _WORKER_STATE['r']
    K = _WORKER_STATE['K']
    q = _WORKER_STATE['q']
    seed0 = _WORKER_STATE['seed0']
    R = _WORKER_STATE['R']
    V_oracle_r = _WORKER_STATE['V_oracle_r']

    rng = np.random.default_rng(seed0 + i_N * n_rep + rep)
    Z_true = rng.multivariate_normal(np.zeros(d), R, size=N)
    U = stats.norm.cdf(Z_true)
    X = np.log(U / (1.0 - U))  # logistic marginal

    res = cas_fit(X, K=K, q=q, r=r, seed=seed0 + rep)
    return (i_N, rep, sin_theta_F(res['V_r'], V_oracle_r))


def run_rate_verification(d=20, r=3, rho=0.7, K=2, q=2,
                          N_values=None, n_rep=20, seed0=500,
                          out_png='rate_plot.png',
                          n_workers=None):
    if N_values is None:
        N_values = np.array([300, 500, 1000, 2000, 4000, 8000])
    if n_workers is None:
        n_workers = min(int(os.environ.get('N_WORKERS',
                                           max(1, (os.cpu_count() or 2) - 1))),
                        len(N_values) * n_rep)

    # Oracle: 3-block correlation R, analytic C = (R^{-1} - I) R (R^{-1} - I)^T
    R = np.eye(d)
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, j] = rho
    Rinv = np.linalg.inv(R)
    C_or = (Rinv - np.eye(d)) @ R @ (Rinv - np.eye(d)).T
    w_or, V_or = np.linalg.eigh(C_or)
    idx = np.argsort(-w_or)
    V_oracle_r = V_or[:, idx[:r]]

    # Build task list and run in parallel
    tasks = [(i, int(N), rep, n_rep)
             for i, N in enumerate(N_values)
             for rep in range(n_rep)]

    from cas.parallel import run_parallel_init

    initargs = (d, r, rho, K, q, seed0, R.tolist(), V_oracle_r.tolist())
    results = run_parallel_init(_run_one_rep, tasks, n_workers,
                                initializer=_init_worker, initargs=initargs,
                                desc='rate-plot cas_fits')

    sinFs = np.zeros((len(N_values), n_rep))
    for i_N, rep, val in results:
        sinFs[i_N, rep] = val

    for i, N in enumerate(N_values):
        print(f"  N={int(N):>5}: mean sin_Theta_F = {sinFs[i].mean():.4f} "
              f"(std {sinFs[i].std():.4f}) over {n_rep} reps")

    # Log-log least-squares slope on mean across reps
    log_N = np.log(N_values)
    log_sin = np.log(sinFs.mean(axis=1))
    slope, intercept = np.polyfit(log_N, log_sin, 1)
    print(f"\nFitted log-log slope: {slope:.3f} (theory predicts -0.5)")

    # --- plot ---
    plt.rcParams.update({
        'font.size':       20.4,
        'axes.labelsize':  20.4,
        'axes.titlesize':  21.2,
        'legend.fontsize': 18.7,
        'xtick.labelsize': 18.7,
        'ytick.labelsize': 18.7,
    })
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    med = np.median(sinFs, axis=1)
    q25 = np.percentile(sinFs, 25, axis=1)
    q75 = np.percentile(sinFs, 75, axis=1)
    ax.errorbar(N_values, med, yerr=[med - q25, q75 - med],
                fmt='o-', capsize=3, label='measured (median, IQR)', color='C0')

    # Theoretical reference N^{-1/2} anchored by geometric mean
    log_c = (log_sin - (-0.5) * log_N).mean()
    ref = np.exp(log_c) * N_values ** (-0.5)
    ax.plot(N_values, ref, '--', color='C3',
            label=r'$\propto N^{-1/2}$ reference')

    # Fitted slope
    ax.plot(N_values, np.exp(intercept) * N_values ** slope, ':', color='C2',
            label=f'fitted slope {slope:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\|\sin\Theta(\hat V_r, V_r)\|_F$')
    # Caption-carrying title: setup details belong in the supplement caption.
    ax.legend(fontsize=15.3, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_png}")
    return N_values, sinFs, slope


if __name__ == '__main__':
    out_png = os.environ.get('RATE_PLOT_OUT',
                              os.path.join(HERE, 'rate_plot.png'))
    run_rate_verification(out_png=out_png)
