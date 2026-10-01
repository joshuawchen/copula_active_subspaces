"""Part I, fig:recovery-row: recovery angle against the reference V_r^C versus N
for CAS, PCA and a random subspace, Examples 1-3.
Reads cache/run2026/{example}/subspace_N{N}.json; writes recovery_row.png.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, 'src'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas.plotstyle import (apply_style, COLOR_CAS, COLOR_PCA, COLOR_RANDOM,
                            TEXTWIDTH_IN)

apply_style()
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7})
plt.rcParams['savefig.bbox'] = 'standard'


CACHE_DIR = os.environ.get('V82_CACHE_DIR', os.path.join(_REPO, 'cache'))
N_LIST = [500, 1000, 2000, 5000, 10000, 20000, 50000]
OUT = os.environ.get('RECOVERY_ROW_OUT', os.path.join(HERE, 'recovery_row.png'))
_keep = os.environ.get('RECOVERY_N_KEEP', '500,2000,10000,20000,50000').strip()
N_KEEP = set(int(x) for x in _keep.split(',')) if _keep else None
YSCALE = os.environ.get('RECOVERY_YSCALE', 'linear')

EXAMPLES = [('banana', 'Example 1'), ('even_fold', 'Example 2'),
            ('conformal_cube', 'Example 3')]
SERIES = [('CAS',         'CAS',    COLOR_CAS,    'o', '-'),
          ('PCA',         'PCA',    COLOR_PCA,    's', '-'),
          ('Uniformly random', 'random', COLOR_RANDOM, '^', '--')]


def _subspace_path(example, N):
    return os.path.join(CACHE_DIR, 'run2026', example, f'subspace_N{N}.json')


def frob_sin_to_deg(s, r):
    return float(np.degrees(np.arcsin(min(s / np.sqrt(r), 1.0))))


def _plot_panel(ax, example, r, show_ylabel, title):
    Ns = [N for N in N_LIST if os.path.exists(_subspace_path(example, N))]
    if N_KEEP:
        Ns = [n for n in Ns if n in N_KEEP]
    data = {}
    for N in Ns:
        with open(_subspace_path(example, N)) as f:
            data[N] = json.load(f)
    for label, method, color, mk, ls in SERIES:
        m, lo, hi = [], [], []
        for N in Ns:
            cell = data[N][method]
            degs = np.array([frob_sin_to_deg(cell[s], r)
                             for s in sorted(cell, key=int)])
            mu, sd = float(degs.mean()), float(degs.std())
            m.append(mu)
            lo.append(max(0.0, mu - sd))
            hi.append(mu + sd)
        ax.fill_between(Ns, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(Ns, m, color=color, marker=mk, ls=ls, lw=1.3, ms=4, label=label)
    ax.set_xscale('log')
    ax.set_xlabel(r'Sample size $N$', fontsize=8)
    ax.set_title(title, fontsize=8)
    if YSCALE == 'log':
        ax.set_yscale('log')
        ax.set_ylim(0.7, 95)
    else:
        ax.set_ylim(0, 72)
    if Ns:
        ax.set_xlim(min(Ns) * 0.85, max(Ns) * 1.18)
    ax.grid(True, which='both', alpha=0.25)
    if show_ylabel:
        ax.set_ylabel(r'Principal angle (deg/direction)', fontsize=8)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Fig 3: three-panel subspace recovery')
    ap.add_argument('--out', default=OUT, help='output path')
    out = ap.parse_args().out

    r = 4
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.4), sharey=True)
    for ax, (ex, title) in zip(axes, EXAMPLES):
        _plot_panel(ax, ex, r, ax is axes[0], title)
    _h, _l = axes[0].get_legend_handles_labels()
    plt.gcf().legend(_h, _l, loc='lower center', ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out}')


if __name__ == '__main__':
    main()
