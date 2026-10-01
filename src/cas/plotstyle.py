"""Figure sizes and fonts of the paper figures (text width 5.0 in)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as _LSC

TEXTWIDTH_IN = 5.0
SHORT_H_IN   = 2.6
MEDIUM_H_IN  = 3.5
TALL_H_IN    = 4.5

# ---- Palettes --------------------------------------------------------------
OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'sky_blue': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermilion': '#D55E00', 'purple': '#CC79A7',
}
OKABE_ITO_CYCLE = ['#D55E00', '#0072B2', '#009E73', '#CC79A7',
                   '#E69F00', '#56B4E9', '#F0E442', '#000000']

COLOR_ORACLE  = '#000000'
COLOR_CAS     = '#D55E00'
COLOR_PCA     = '#009E73'
COLOR_GAUSS   = '#0072B2'
COLOR_PRODUCT = '#CC79A7'
COLOR_RANDOM  = '#888888'

METHOD_COLORS = {
    'oracle': COLOR_ORACLE, 'true': COLOR_ORACLE,
    'cas': COLOR_CAS,
    'gauss': COLOR_GAUSS, 'gauss_copula': COLOR_GAUSS, 'gaussian': COLOR_GAUSS,
    'product': COLOR_PRODUCT, 'product_marg': COLOR_PRODUCT, 'pom': COLOR_PRODUCT,
    'pca': COLOR_PCA, 'pca_vr': COLOR_PCA, 'pca_r4': COLOR_PCA,
    'random': COLOR_RANDOM,
}

def _tint_to(color, name, start_alpha=0.25):
    """Light tint of color at start_alpha -> full color at 1.0. Avoids pure
    white at low density so contours stay readable."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(color))
    light = (1 - start_alpha) * np.ones(3) + start_alpha * rgb
    return _LSC.from_list(name, [(0.0, light), (1.0, rgb)])

CMAP_ORACLE = _tint_to('#000000',     'oracle_cmap', start_alpha=0.18)
CMAP_CAS    = _tint_to(COLOR_CAS,     'cas_cmap',    start_alpha=0.30)
CMAP_PCA    = _tint_to(COLOR_PCA,     'pca_cmap',    start_alpha=0.30)
CMAP_GAUSS  = _tint_to(COLOR_GAUSS,   'gauss_cmap',  start_alpha=0.30)
CMAP_POM    = _tint_to(COLOR_PRODUCT, 'pom_cmap',    start_alpha=0.30)

METHOD_CMAPS = {
    'oracle': CMAP_ORACLE, 'true': CMAP_ORACLE,
    'cas': CMAP_CAS,
    'pca': CMAP_PCA, 'pca_vr': CMAP_PCA, 'pca_r4': CMAP_PCA,
    'gauss': CMAP_GAUSS, 'gauss_copula': CMAP_GAUSS, 'gaussian': CMAP_GAUSS,
    'pom': CMAP_POM, 'product': CMAP_POM, 'product_marg': CMAP_POM,
}

CAS_VARIANT_COLORS = {
    (2, 2): '#CC79A7', (3, 3): '#E69F00', (4, 2): COLOR_CAS, (5, 2): '#56B4E9',
}

def cas_ramp(n):
    cmap = plt.get_cmap('plasma')
    return [cmap(x) for x in np.linspace(0.15, 0.75, n)]

def apply_style():
    """Strict uniform style. Generate every figure at width=TEXTWIDTH_IN."""
    plt.rcParams.update({
        'text.usetex':      False,
        'mathtext.fontset': 'cm',
        'font.family':      'serif',
        'font.serif':       ['cmr10', 'Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'axes.formatter.use_mathtext': True,
        'font.size':        9,
        'axes.labelsize':   9,
        'axes.titlesize':   10,
        'axes.titleweight': 'normal',
        'legend.fontsize':  8,
        'xtick.labelsize':  8,
        'ytick.labelsize':  8,
        'axes.grid':        True,
        'grid.alpha':       0.25,
        'grid.linewidth':   0.4,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.7,
        'lines.linewidth':   1.3,
        'lines.markersize':  4.5,
        'lines.markeredgewidth': 0.6,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.02,
        'xtick.major.pad':   2,
        'ytick.major.pad':   2,
    })
