"""Map rendering: composite city maps, single layers, layer panels, fields."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch

from ..grid.layers import spec_for
from . import style

ANCHOR_MARKERS = {"cbd": "*", "subcenter": "o", "airport": "^", "port": "s", "station": "D",
                  "university": "p", "industrial": "h", "generic": "o"}


def _rgb(c):
    return np.array(to_rgb(c))


def base_rgb(grid) -> np.ndarray:
    """Hillshaded land, water and protected land as an (H, W, 3) image."""
    shade = grid.derive("hillshade") if np.ptp(grid["elevation"]) > 0 else np.full(grid.shape, 0.7)
    img = _rgb(style.LAND)[None, None, :] * (0.75 + 0.3 * shade[..., None])
    ex = np.clip(grid["excluded"], 0, 1)[..., None]
    img = img * (1 - 0.25 * ex) + _rgb(style.PROTECTED) * 0.25 * ex
    img[grid["water"] > 0] = _rgb(style.WATER)
    return np.clip(img, 0, 1)


def render_map(ax, grid, urban=None, roads=None, urbanized_at=None, year=None,
               start_year=None, end_year=None, phi=None, anchors=True,
               title: str | None = None, show_legend: bool = True):
    """Composite map. Arrays default to the grid's current layers, so the same
    function draws both the live grid and recorded history frames."""
    urban = grid["urban"] if urban is None else urban
    roads = grid["roads"] if roads is None else roads
    urbanized_at = grid["urbanized_at"] if urbanized_at is None else urbanized_at
    year = grid.year if year is None else year
    start_year = np.nanmin(urbanized_at) if start_year is None else start_year
    end_year = max(year, start_year + 1) if end_year is None else end_year

    img = base_rgb(grid)
    for cls, col in sorted(style.ROAD_COLORS.items()):
        img[roads == cls] = _rgb(col)
    u = urban > 0
    old = u & (urbanized_at <= start_year)
    img[old] = _rgb(style.URBAN_OLD)
    new = u & ~old
    if new.any():
        t = (urbanized_at[new] - start_year) / max(end_year - start_year, 1e-9)
        img[new] = style.GROWTH(np.clip(t, 0, 1))[:, :3]
    # Draw major roads over urban so the network stays legible.
    for cls in (2, 3):
        m = (roads == cls) & u
        img[m] = img[m] * 0.4 + _rgb(style.ROAD_COLORS[cls]) * 0.6
    ax.imshow(img, interpolation="nearest")

    if phi is not None and np.ptp(phi) > 0:
        levels = np.linspace(phi.min(), phi.max(), 9)[1:-1]
        ax.contour(phi, levels=levels, colors=style.INK_2, linewidths=0.6, alpha=0.6)

    if anchors:
        draw_anchors(ax, grid, year)
    style.map_axes(ax)
    ax.set_title(title or f"{grid.name} — {year:g}", loc="left")
    if show_legend:
        handles = [Patch(color=style.URBAN_OLD, label=f"Urban by {start_year:g}"),
                   Patch(color=style.GROWTH(0.15), label="Earlier growth"),
                   Patch(color=style.GROWTH(0.85), label="Recent growth"),
                   Patch(color=style.WATER, label="Water"),
                   Patch(color=style.ROAD_COLORS[3], label="Highway")]
        if (grid["excluded"] > 0).any():
            handles.append(Patch(color="#a9c9a0", label="Protected"))
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85,
                  frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
    return ax


def draw_anchors(ax, grid, year=None):
    year = grid.year if year is None else year
    for a in grid.anchors:
        m = ANCHOR_MARKERS.get(a.kind, "o")
        s = a.strength(year)
        x, y = a.position
        if a.is_open(year):
            ax.plot(x, y, m, ms=11, mfc=style.INK, mec=style.SURFACE, mew=1.5)
            status = ""
        elif s > 0:
            ax.plot(x, y, m, ms=11, mfc=style.SURFACE, mec=style.INK, mew=1.8)
            status = f" (opens {a.open_year:g})"
        else:
            ax.plot(x, y, m, ms=9, mfc="none", mec=style.INK_MUTED, mew=1.0, ls="none")
            status = " (planned)"
        ax.annotate(a.name + status, (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8, color=style.INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.75))


def render_layer(ax, grid, name: str, cmap=None, colorbar: bool = True, **params):
    arr = grid.derive(name, **params) if params else grid[name]
    arr = np.asarray(arr, dtype=float)
    spec = spec_for(name)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros(grid.shape)
    im = ax.imshow(np.where(finite, arr, np.nan), cmap=cmap or spec.cmap, interpolation="nearest")
    style.map_axes(ax)
    ax.set_title(name, loc="left", fontsize=9)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    return im


def render_layers(grid, names=None, cols: int = 4, size: float = 3.2):
    """Panel of many layers (base and derived) for inspecting a city."""
    style.apply()
    names = names or (grid.layer_names() + ["slope", "road_distance", "urban_density", "center_distance"])
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 0.95))
    for ax, n in zip(np.atleast_1d(axes).ravel(), names):
        try:
            render_layer(ax, grid, n)
        except Exception as e:  # keep the panel even if one layer fails
            ax.text(0.5, 0.5, f"{n}\n{e}", ha="center", va="center", fontsize=7, wrap=True)
            style.map_axes(ax)
    for ax in np.atleast_1d(axes).ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{grid.name} — layers ({grid.year:g})", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig


def render_field(ax, grid, phi=None, title="Anchor potential Φ"):
    """Signed potential on a diverging scale (grey = no influence)."""
    phi = grid["anchor_potential"] if phi is None else phi
    vmax = max(float(np.abs(phi).max()), 1e-6)
    shade = grid.derive("hillshade") if np.ptp(grid["elevation"]) > 0 else None
    if shade is not None:
        ax.imshow(shade, cmap="Greys_r", alpha=1.0, interpolation="nearest")
    im = ax.imshow(phi, cmap=style.DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), alpha=0.85,
                   interpolation="nearest")
    water = np.ma.masked_where(grid["water"] == 0, grid["water"])
    ax.imshow(water, cmap=plt.matplotlib.colors.ListedColormap([style.WATER]), interpolation="nearest")
    draw_anchors(ax, grid)
    style.map_axes(ax)
    ax.set_title(title, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Φ  (attraction > 0, repulsion < 0)", fontsize=8)
    return im


def render_probability(ax, grid, prob, title="Probability of being urban", year=None):
    """Monte Carlo urbanisation probability; existing urban shown in ink."""
    img = base_rgb(grid)
    ax.imshow(img, interpolation="nearest")
    existing = grid["urban"] > 0
    p = np.ma.masked_where((prob <= 0.02) | existing, prob)
    im = ax.imshow(p, cmap=style.SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~existing, existing),
              cmap=plt.matplotlib.colors.ListedColormap([style.URBAN_OLD]), interpolation="nearest")
    draw_anchors(ax, grid, year)
    style.map_axes(ax)
    ax.set_title(title, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("P(urban)", fontsize=8)
    ax.legend(handles=[Patch(color=style.URBAN_OLD, label=f"Urban at start ({grid.year:g})")],
              loc="lower right", fontsize=7, frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
    return im


def render_comparison(ax, grid, initial, observed, simulated, title=None):
    """Change-agreement map: hits, misses, false alarms (Pontius FoM components)."""
    i, o, s = initial > 0, observed > 0, simulated > 0
    cat = np.full(grid.shape, -1)
    cat[i] = 0
    cat[(o & ~i) & (s & ~i)] = 1
    cat[(o & ~i) & ~s] = 2
    cat[~(o & ~i) & (s & ~i)] = 3
    ax.imshow(base_rgb(grid), interpolation="nearest")
    ax.imshow(np.ma.masked_where(cat < 0, cat), cmap=style.COMPARE, vmin=-0.5, vmax=3.5,
              interpolation="nearest")
    style.map_axes(ax)
    ax.set_title(title or "Simulated vs observed change", loc="left")
    labels = ["Already urban", "Hit (both grew)", "Miss (observed only)", "False alarm (sim only)"]
    ax.legend(handles=[Patch(color=style.COMPARE(k), label=l) for k, l in enumerate(labels)],
              loc="lower right", fontsize=7, frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
