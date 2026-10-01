"""Matplotlib style of the paper figures. apply_paper_style() uses LaTeX when it
is available (CAS_NO_LATEX=1 forces mathtext); the module also defines the
shared palette and axis labels.
"""
from __future__ import annotations

import os
import shutil

import matplotlib

# ---------------------------------------------------------------------------
# Detect whether real LaTeX is usable
# ---------------------------------------------------------------------------
def _latex_available() -> bool:
    if os.environ.get("CAS_NO_LATEX") == "1":
        return False
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    # Probe that cm-super is installed (type1ec.sty)
    import subprocess
    try:
        r = subprocess.run(
            ["kpsewhich", "type1ec.sty"], capture_output=True, text=True, timeout=2,
        )
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


LATEX_AVAILABLE: bool = _latex_available()


# ---------------------------------------------------------------------------
# Palette (Okabe–Ito)
# ---------------------------------------------------------------------------
# Method colors used across Fig 4 / 5 / 6 / SM3
COLOR_CAS      = "#D55E00"   # vermillion (Okabe-Ito)
COLOR_PCA      = "#009E73"   # bluish green (Okabe-Ito)
COLOR_GAUSS    = "#0072B2"   # blue (Okabe-Ito)
COLOR_POM      = "#CC79A7"   # reddish purple (Okabe-Ito)
COLOR_TRUE     = "#000000"   # black for the true posterior / oracle reference
COLOR_RANDOM   = "#888888"   # neutral grey for Haar-random subspaces


# ---------------------------------------------------------------------------
# Shared text labels (raw LaTeX strings — work in both modes)
# ---------------------------------------------------------------------------
KL_SYMBOL        = r"$D_{\mathrm{KL}}$"
KL_LABEL_LOWER   = r"$D_{\mathrm{KL}} \downarrow$"
KL_LABEL_FULL    = r"$D_{\mathrm{KL}}(\pi \,\|\, \hat\pi)\ \downarrow$"
KL_NOISE_LABEL   = r"Noise $D_{\mathrm{KL}}(\pi_\eta \,\|\, \hat\pi_\eta)\ \downarrow$"
KL_POST_LABEL    = r"Posterior $D_{\mathrm{KL}}(\pi(\cdot\,|\,y) \,\|\, \hat\pi(\cdot\,|\,y))\ \downarrow$"


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------
def apply_paper_style() -> None:
    """Set ``matplotlib.rcParams`` for paper-grade typography."""
    import matplotlib.pyplot as plt

    base = {
        "font.family":      "serif",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.linewidth":    0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    }
    if LATEX_AVAILABLE:
        base["text.usetex"] = True
        base["font.serif"] = ["Computer Modern Roman"]
        base["text.latex.preamble"] = (
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}"
        )
    else:
        # Mathtext fallback — use STIX-like glyphs that match LaTeX visually
        base["text.usetex"] = False
        base["mathtext.fontset"] = "cm"   # Computer Modern in mathtext
    plt.rcParams.update(base)


# ---------------------------------------------------------------------------
# Shared figure conventions.
#
# Every figure in both papers places its legend the same way: above the axes,
# in a row, outside the data. Legends drawn inside the axes were sitting on
# top of bars and curves, and each figure had chosen its own corner, so no two
# panels read alike. The panel letter goes in the top-left of the axes rather
# than in a title, since a title would compete with the legend for the space
# above the frame.
#
# Labels passed to these must render under BOTH renderers: apply_paper_style
# switches text.usetex on or off depending on the environment, so use plain
# text and basic mathtext only. LaTeX-only commands (\textsc, \text, custom
# macros such as \tr) render in usetex mode and fail in mathtext mode.
# ---------------------------------------------------------------------------
def top_legend(ax, ncol, handles=None, labels=None, fontsize=6.4, pad=1.01):
    """Legend above the axes, in a row, outside the data."""
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, pad), ncol=ncol,
              frameon=False, fontsize=fontsize, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0)
    if handles is None:
        ax.legend(**kw)
    else:
        ax.legend(handles, labels, **kw)


def panel_tag(ax, text, fontsize=8):
    """Panel letter in the top-left of the axes, in place of a title."""
    ax.text(0.025, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold")
