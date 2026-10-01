"""Part I, fig:pedagogical: two-dimensional illustration of Stage 1 (samples
with the estimated direction, the score field, the spectrum of C_hat).
Writes pedagogical_2d.png.
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (os.path.join(_REPO, 'src'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cas import cas_fit, hermite_norm  # noqa: E402


# ---------------------------------------------------------------
# Banana copula target
# ---------------------------------------------------------------

def sample_banana_2d(N, a, theta_rot, rng):
    """2D rotated banana copula with logistic marginals."""
    G = rng.standard_normal((N, 2))
    W1 = G[:, 0]
    W2 = G[:, 1] + a * (W1 ** 2 - 1.0)
    W = np.column_stack([W1, W2])
    c, s = np.cos(theta_rot), np.sin(theta_rot)
    Q = np.array([[c, -s], [s, c]])
    Z_latent = W @ Q.T
    from scipy.stats import norm
    U = norm.cdf(Z_latent)
    U = np.clip(U, 1e-12, 1.0 - 1e-12)
    eta = np.log(U / (1.0 - U))
    return eta, Q


# ---------------------------------------------------------------
# Score-field evaluator
# ---------------------------------------------------------------

def eval_score_field(Z_grid, theta, A, K):
    M = Z_grid.shape[0]
    H_per_dim = [hermite_norm(Z_grid[:, i], K) for i in range(2)]
    tau = np.zeros((M, 2), dtype=float)
    for theta_alpha, (supp, degs) in zip(theta, A):
        if abs(theta_alpha) == 0.0:
            continue
        for j, d_j in zip(supp, degs):
            coeff = math.sqrt(d_j)
            prod_others = np.ones(M, dtype=float)
            for k, d_k in zip(supp, degs):
                if k == j:
                    continue
                prod_others *= H_per_dim[k][:, d_k]
            deriv_factor = H_per_dim[j][:, d_j - 1]
            tau[:, j] += theta_alpha * coeff * prod_others * deriv_factor
    return tau


# ---------------------------------------------------------------
# Arrow helper
# ---------------------------------------------------------------

def draw_arrow(ax, start, end, color, lw=2.6, ls='-', alpha=1.0,
               zorder=5, arrowsize=22):
    arrow = FancyArrowPatch(
        tuple(start), tuple(end),
        arrowstyle='-|>',
        color=color, lw=lw, linestyle=ls, alpha=alpha, zorder=zorder,
        mutation_scale=arrowsize, shrinkA=0, shrinkB=0,
        joinstyle='round', capstyle='round',
    )
    arrow.set_facecolor(color)
    arrow.set_edgecolor(color)
    ax.add_patch(arrow)


# ---------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------

def main():
    # ---- config ----
    N = 8000
    K = 4
    q = 2
    a = 0.5
    theta_rot = np.deg2rad(35.0)
    fit_seed = 0
    sample_seed = 1

    rng = np.random.default_rng(sample_seed)
    eta, Q = sample_banana_2d(N, a, theta_rot, rng)

    result = cas_fit(eta, K=K, q=q, r=2, seed=fit_seed, return_all=True)
    Z = result['Z']
    V_r = result['V_r']
    eigvals = result['eigvals_top']
    theta = result['theta']
    A = result['A']

    v_star = Q[:, 0]
    v_star_perp = Q[:, 1]
    v1_hat = V_r[:, 0]
    if np.dot(v_star, v1_hat) < 0:
        v1_hat = -v1_hat

    cos_angle = float(np.clip(abs(np.dot(v_star, v1_hat)), 0.0, 1.0))
    angle_deg = np.degrees(np.arccos(cos_angle))
    ratio = float(eigvals[0] / max(eigvals[1], 1e-12))

    print(f"  angle(v_hat, v*)  = {angle_deg:.2f} deg")
    print(f"  eigvals           = {eigvals}")
    print(f"  lambda_1/lambda_2 = {ratio:.2f}")

    # ---- palette ----
    # Samples: a soft warm grey so the eye doesn't get distracted by them
    # and the colored arrows pop. Method colors are unified with Fig 4-6
    # (Okabe–Ito): CAS in vivid orange #E69F00.
    from cas.paperstyle import COLOR_CAS, apply_paper_style
    C_SAMPLE = '#BFBFBF'       # neutral medium-light grey for samples
    C_ORACLE = '#111111'       # black for true direction v*
    C_PERP   = '#8A8A8A'       # medium grey for v*_perp
    C_HAT    = COLOR_CAS       # CAS orange — matches Fig 4-6
    C_ARROWS = '#606060'       # uniform grey for score-field quiver
    C_GRID   = '#DDDDDD'

    apply_paper_style()
    plt.rcParams.update({
        'font.size':        9,
        'axes.labelsize':   10,
        'axes.titlesize':   10,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  9,
    })

    fig = plt.figure(figsize=(10.0, 3.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.74],
                  wspace=0.15, left=0.050, right=0.985, top=0.90, bottom=0.15)

    # =========================================================
    # PANEL (a): samples + oracle basis + v_hat
    # =========================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(Z[:, 0], Z[:, 1], s=3.0, alpha=0.45, color=C_SAMPLE,
                 edgecolor='none', rasterized=True, zorder=1)

    # Scale oracle basis arrows by sqrt(eigenvalues). Max length -> L_ref.
    L_ref = 2.2
    lam_sqrt = np.sqrt(np.array([eigvals[0], eigvals[1]]))
    lam_scale = lam_sqrt / lam_sqrt.max() * L_ref

    origin = np.array([0.0, 0.0])
    ax_a.axhline(0, color=C_GRID, lw=0.7, zorder=0)
    ax_a.axvline(0, color=C_GRID, lw=0.7, zorder=0)

    end_star = origin + lam_scale[0] * v_star
    draw_arrow(ax_a, origin, end_star, C_ORACLE, lw=2.8, ls='-',
               zorder=5, arrowsize=22)

    end_perp = origin + lam_scale[1] * v_star_perp
    draw_arrow(ax_a, origin, end_perp, C_PERP, lw=2.4, ls='-',
               zorder=4, arrowsize=18)

    # v_hat translated by 0.5 in v*_perp direction so it sits above v*
    translate = 0.55 * v_star_perp
    end_hat = translate + L_ref * v1_hat
    start_hat = translate - 0.15 * v1_hat
    draw_arrow(ax_a, start_hat, end_hat, C_HAT, lw=3.0, ls='-',
               zorder=6, arrowsize=22)

    # Labels near arrow tips
    ax_a.text(end_star[0] + 0.22, end_star[1] - 0.22,
              r'$v^\star$', color=C_ORACLE, fontsize=11, fontweight='bold',
              zorder=7, ha='left', va='center')
    ax_a.text(end_perp[0] - 0.22, end_perp[1] + 0.22,
              r'$v^\star_\perp$', color=C_PERP, fontsize=10,
              zorder=7, ha='right', va='center')
    ax_a.text(end_hat[0] - 0.22, end_hat[1] + 0.32,
              r'$\hat v_1$', color=C_HAT, fontsize=11, fontweight='bold',
              zorder=7, ha='right', va='center')

    # Subtle angle annotation in the corner (matches original's style)
    ax_a.text(0.04, 0.04, fr'$\angle(\hat v_1, v^\star) = {angle_deg:.1f}^\circ$',
              transform=ax_a.transAxes, fontsize=9,
              ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#CCCCCC', alpha=0.92))

    ax_a.set_xlim(-3.3, 3.3)
    ax_a.set_ylim(-3.3, 3.3)
    ax_a.set_aspect('equal')
    ax_a.set_xlabel(r'$z_1$')
    ax_a.set_ylabel(r'$z_2$')
    ax_a.set_title(r'(a) Samples $Z$, true $v^\star$, $v^\star_\perp$, recovered $\hat v_1$',
                   fontsize=10)
    ax_a.grid(True, alpha=0.22, zorder=0)

    # =========================================================
    # PANEL (b): score field — single-color grey family for contrast
    # =========================================================
    ax_b = fig.add_subplot(gs[0, 1])
    # Light scatter as context, under arrows
    ax_b.scatter(Z[:, 0], Z[:, 1], s=2.2, alpha=0.15, color=C_SAMPLE,
                 edgecolor='none', rasterized=True, zorder=1)

    grid_n = 18
    grid_lim = 3.0
    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    XG, YG = np.meshgrid(xs, ys)
    Zg = np.column_stack([XG.ravel(), YG.ravel()])
    tau_vals = eval_score_field(Zg, theta, A, K)
    UG = tau_vals[:, 0].reshape(XG.shape)
    VG = tau_vals[:, 1].reshape(XG.shape)
    magnitude = np.hypot(UG, VG)
    # Cap magnitude at 95th percentile to prevent corner blow-up dominating the scale
    mag_cap = np.percentile(magnitude, 95)
    scale = np.where(magnitude > mag_cap, mag_cap / np.maximum(magnitude, 1e-8), 1.0)
    UG_clip = UG * scale
    VG_clip = VG * scale

    # Uniform grey arrows (quiver)
    ax_b.quiver(XG, YG, UG_clip, VG_clip,
                color=C_ARROWS, alpha=0.75,
                scale=None, scale_units='xy', angles='xy',
                width=0.004, headwidth=4.0, headlength=5.0, headaxislength=4.5,
                zorder=3)

    # v_hat prominent on top
    end_hat_b = L_ref * v1_hat
    draw_arrow(ax_b, -end_hat_b, end_hat_b, C_HAT, lw=3.2, ls='-',
               zorder=6, arrowsize=22)
    ax_b.text(end_hat_b[0] - 0.22, end_hat_b[1] + 0.32,
              r'$\hat v_1$', color=C_HAT, fontsize=11, fontweight='bold',
              zorder=7, ha='right', va='center')

    ax_b.set_xlim(-grid_lim - 0.3, grid_lim + 0.3)
    ax_b.set_ylim(-grid_lim - 0.3, grid_lim + 0.3)
    ax_b.set_aspect('equal')
    ax_b.set_xlabel(r'$z_1$')
    ax_b.set_ylabel(r'$z_2$')
    ax_b.set_title(r'(b) Recovered score field $\hat\tau(z) = \nabla g_{\hat\theta}(z)$',
                   fontsize=10)
    ax_b.grid(True, alpha=0.22, zorder=0)

    # =========================================================
    # PANEL (c): spectrum — colors MATCH the vectors
    # =========================================================
    ax_c = fig.add_subplot(gs[0, 2])
    bar_colors = [C_HAT, C_PERP]       # lambda_1 matches v_hat; lambda_2 matches v*_perp
    bar_edges  = ['#A6460B', '#5A5A5A']
    bars = ax_c.bar([1, 2], eigvals[:2], color=bar_colors,
                    edgecolor=bar_edges, linewidth=1.4, width=0.64, zorder=3)
    for b, v in zip(bars, eigvals[:2]):
        ax_c.text(b.get_x() + b.get_width() / 2.0,
                  b.get_height() + max(eigvals[:2]) * 0.015,
                  f'{v:.3f}',
                  ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Ratio bracket spanning the two bars
    y_top = max(eigvals[:2]) * 1.26
    y_tick = max(eigvals[:2]) * 0.04
    ax_c.plot([1, 1, 2, 2], [y_top - y_tick, y_top, y_top, y_top - y_tick],
              color='black', lw=1.1, zorder=4, solid_capstyle='round')
    ax_c.text(1.5, y_top * 1.04,
              fr'$\hat\lambda_1/\hat\lambda_2 \approx {ratio:.1f}$',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels([r'$\hat\lambda_1$', r'$\hat\lambda_2$'], fontsize=9)
    ax_c.set_ylabel(r'Eigenvalue of $\hat C$', fontsize=10)
    ax_c.set_title(r'(c) Spectrum of $\hat C$ (rank-$1$-like)', fontsize=10)
    ax_c.set_ylim(0, max(eigvals[:2]) * 1.48)
    ax_c.grid(True, axis='y', alpha=0.28, zorder=0)
    ax_c.set_axisbelow(True)

    out_path = os.environ.get(
        'PEDAGOGICAL_OUT', os.path.join(HERE, 'pedagogical_2d.png'))
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == '__main__':
    main()
