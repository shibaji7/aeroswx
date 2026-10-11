"""Matplotlib style for the project (deep blue / golden yellow / white / light grey)."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

DEEP_BLUE = "#003366"
GOLDEN_YELLOW = "#FFCC00"
WHITE = "#FFFFFF"
LIGHT_GREY = "#F0F0F0"
INK = "#1F2933"  # text colour: never a series colour
MID_BLUE = "#5C7FA3"
DARK_GOLD = "#8A6D00"  # yellow-family colour with enough contrast to draw thin lines on white

PALETTE = {
    "deep_blue": DEEP_BLUE,
    "golden_yellow": GOLDEN_YELLOW,
    "white": WHITE,
    "light_grey": LIGHT_GREY,
}

#: sequential colormap, light grey -> golden yellow -> deep blue (ordered by lightness)
SEQ_CMAP = LinearSegmentedColormap.from_list("swe_seq", [LIGHT_GREY, GOLDEN_YELLOW, DEEP_BLUE])
#: single-hue (blue) sequential map for magnitudes
BLUE_CMAP = LinearSegmentedColormap.from_list("swe_blue", ["#E6EDF5", MID_BLUE, DEEP_BLUE])


def style_path() -> str:
    return str(resources.files("swe_airline_delay.plotting").joinpath("swe.mplstyle"))


def use_style() -> None:
    """Activate the project style globally (default for all figure functions)."""
    plt.style.use(style_path())


@contextmanager
def style_context():
    with plt.style.context(style_path()):
        yield


def yellow_outline() -> list:
    """Thin deep-blue stroke so golden-yellow lines/markers stay visible on white."""
    return [patheffects.withStroke(linewidth=3.2, foreground=DEEP_BLUE, alpha=0.9)]


def finish(fig, path=None):
    """Save (if ``path``) and return ``fig``."""
    if path:
        fig.savefig(path)
    return fig


mpl.rcParams["axes.unicode_minus"] = False
