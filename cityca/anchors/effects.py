"""Physical effects applied when an anchor opens.

Opening an airport doesn't only create attraction: the site itself is built
(``footprint_radius``) and a link road ties it into the network
(``access_road``). These are applied once by the simulation engine.
"""

from __future__ import annotations

import numpy as np

from ..grid.raster import circle_mask, polyline_mask
from .anchor import Anchor


def build_footprint(grid, anchor: Anchor) -> int:
    if anchor.footprint_radius <= 0:
        return 0
    mask = circle_mask(grid.shape, anchor.position, anchor.footprint_radius) & (grid["water"] == 0)
    return grid.urbanize(mask)


def build_access_road(grid, anchor: Anchor) -> bool:
    """Straight link from the anchor to the nearest existing road cell."""
    if anchor.access_road <= 0:
        return False
    roads = grid["roads"]
    rr, cc = np.nonzero(roads > 0)
    if rr.size == 0:
        return False
    r0, c0 = anchor.rc
    k = np.argmin((rr - r0) ** 2 + (cc - c0) ** 2)
    line = polyline_mask(grid.shape, [anchor.position, (cc[k], rr[k])]) & (grid["water"] == 0)
    roads[line] = np.maximum(roads[line], anchor.access_road)
    grid.touch("roads")
    return True


def activate(grid, anchor: Anchor) -> dict:
    return {"footprint_cells": build_footprint(grid, anchor),
            "access_road": build_access_road(grid, anchor)}
