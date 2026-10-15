"""Shared colours and matplotlib styling (validated reference palette)."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"

# Categorical slots, fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING_MID = "#f0efec"

# Map layers
LAND = "#efeee9"
WATER = "#b7d3f6"
PROTECTED = "#008300"
ROAD_COLORS = {1: "#b9b8b1", 2: "#7d7c76", 3: "#2e2e2c"}
URBAN_OLD = "#52514e"

SEQUENTIAL = LinearSegmentedColormap.from_list("cityca_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "cityca_div", ["#0d366b", "#2a78d6", "#86b6ef", DIVERGING_MID, "#f2a3a2", "#e34948", "#8f1f1f"])
# Urbanisation year: dark (early) -> light (recent) in one hue (orange).
GROWTH = LinearSegmentedColormap.from_list("cityca_growth", ["#7a2c0c", "#c2511f", "#eb6834", "#f5a77f", "#fbd5bf"])
COMPARE = ListedColormap(["#d9d8d2", SERIES[0], SERIES[1], SERIES[2]])  # persist, hit, miss, false alarm


def apply() -> None:
    """Recessive axes, thin lines, consistent typography."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "font.size": 9,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def map_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
