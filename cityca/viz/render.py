"""Map rendering: composite city maps, single layers, layer panels, fields."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch

from ..grid.layers import spec_for
from . import style

ANCHOR_MARKERS = {"cbd": "*", "subcenter": "o", "airport": "^", "port": "s", "station": "D",
                  "university": "p", "industrial": "h", "tech_park": "P", "generic": "o"}


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


def density_rgba(density: np.ndarray, full_at: float = 0.35, smooth: float = 1.0) -> np.ndarray:
    """Heatmap colours plus an alpha that fades in from bare land (0) to opaque.

    ``smooth`` (cells) applies a light Gaussian blur for display only, giving a
    continuous heatmap instead of hard cell edges; 0 draws raw cells.
    """
    d = np.clip(np.nan_to_num(density), 0.0, 1.0)
    if smooth > 0:
        from scipy import ndimage
        d = ndimage.gaussian_filter(d, smooth)
    rgba = style.DENSITY(d)
    rgba[..., 3] = np.clip(d / full_at, 0.0, 1.0)
    return rgba


def render_map(ax, grid, urban=None, roads=None, urbanized_at=None, year=None,
               start_year=None, end_year=None, phi=None, anchors=True,
               title: str | None = None, show_legend: bool = True,
               density=None, mode: str = "auto", colorbar: bool = True,
               smooth: float = 1.0):
    """Composite map. Arrays default to the grid's current layers, so the same
    function draws both the live grid and recorded history frames.

    ``mode``: ``density`` draws built-up density as a heatmap (the default when
    a density layer exists); ``age`` colours urban cells by when they were
    urbanised.
    """
    urban = grid["urban"] if urban is None else urban
    roads = grid["roads"] if roads is None else roads
    year = grid.year if year is None else year
    if density is None and "density" in grid.layers:
        density = grid["density"]
    if mode == "auto":
        mode = "density" if density is not None else "age"

    img = base_rgb(grid)
    for cls, col in sorted(style.ROAD_COLORS.items()):
        img[roads == cls] = _rgb(col)
    u = urban > 0

    if mode == "density":
        rgba = density_rgba(density, smooth=smooth)
        a = rgba[..., 3:4]
        img = img * (1 - a) + rgba[..., :3] * a
        over = u
    elif mode == "age":
        urbanized_at = grid["urbanized_at"] if urbanized_at is None else urbanized_at
        start_year = np.nanmin(urbanized_at) if start_year is None else start_year
        end_year = max(year, start_year + 1) if end_year is None else end_year
        old = u & (urbanized_at <= start_year)
        img[old] = _rgb(style.URBAN_OLD)
        new = u & ~old
        if new.any():
            t = (urbanized_at[new] - start_year) / max(end_year - start_year, 1e-9)
            img[new] = style.GROWTH(np.clip(t, 0, 1))[:, :3]
        over = u
    else:
        raise ValueError(f"Unknown map mode '{mode}'")
    # Draw major roads over built-up land so the network stays legible.
    for cls in (2, 3):
        m = (roads == cls) & over
        img[m] = img[m] * 0.55 + _rgb(style.ROAD_COLORS[cls]) * 0.45
    img[grid["water"] > 0] = _rgb(style.WATER)
    ax.imshow(img, interpolation="nearest")

    if phi is not None and np.ptp(phi) > 0:
        levels = np.linspace(phi.min(), phi.max(), 9)[1:-1]
        ax.contour(phi, levels=levels, colors=style.INK_2, linewidths=0.6, alpha=0.6)

    if anchors:
        draw_anchors(ax, grid, year)
    style.map_axes(ax)
    ax.set_title(title or f"{grid.name} — {year:g}", loc="left")
    if mode == "density" and colorbar:
        density_colorbar(ax)
    if show_legend:
        if mode == "age":
            handles = [Patch(color=style.URBAN_OLD, label=f"Urban by {start_year:g}"),
                       Patch(color=style.GROWTH(0.15), label="Earlier growth"),
                       Patch(color=style.GROWTH(0.85), label="Recent growth")]
        else:
            handles = []
        handles += [Patch(color=style.WATER, label="Water"),
                    Patch(color=style.ROAD_COLORS[3], label="Highway")]
        if (grid["excluded"] > 0).any():
            handles.append(Patch(color="#a9c9a0", label="Protected"))
        if any(a.kind == "station" for a in grid.anchors):
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker="D", ls="none", ms=4, mfc=style.INK,
                                  mec=style.SURFACE, label="Metro station"))
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85,
                  frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
    return ax


def density_colorbar(ax, cax=None, label="Built-up density (share of cell built)"):
    """Colour bar for the density heatmap (0 = bare land, 1 = fully built)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=style.DENSITY)
    if cax is None:
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    else:
        cb = plt.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def draw_anchors(ax, grid, year=None, quiet_kinds=("station",)):
    """Anchor markers; ``quiet_kinds`` get small unlabelled markers (e.g. 80 stations)."""
    year = grid.year if year is None else year
    for a in grid.anchors:
        m = ANCHOR_MARKERS.get(a.kind, "o")
        if a.kind in quiet_kinds:
            x, y = a.position
            if a.is_open(year):
                ax.plot(x, y, m, ms=4, mfc=style.INK, mec=style.SURFACE, mew=0.6)
            elif a.strength(year) > 0:
                ax.plot(x, y, m, ms=4, mfc=style.SURFACE, mec=style.INK, mew=0.8)
            continue
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


def render_density_comparison(axes, grid, observed, simulated, title_obs=None, title_sim=None,
                              year=None):
    """Three panels: observed density, simulated density, simulated − observed."""
    from matplotlib.cm import ScalarMappable
    year = grid.year if year is None else year
    for ax, d, t in ((axes[0], observed, title_obs or f"Observed density {year:g}"),
                     (axes[1], simulated, title_sim or f"Simulated density {year:g}")):
        render_map(ax, grid, urban=d >= 0.25, density=d, mode="density", year=year,
                   title=t, show_legend=False, colorbar=False, anchors=False)
    density_colorbar(axes[1])
    ax = axes[2]
    diff = np.where(grid["water"] > 0, np.nan, simulated - observed)
    ax.imshow(base_rgb(grid), interpolation="nearest")
    show = np.ma.masked_where(~np.isfinite(diff) | (np.abs(np.nan_to_num(diff)) < 0.05), diff)
    im = ax.imshow(show, cmap=style.DIVERGING, norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
    style.map_axes(ax)
    ax.set_title("Simulated − observed density", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Δ density (red = model too dense)", fontsize=8)
