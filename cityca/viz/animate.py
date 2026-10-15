"""Animations of recorded simulations (MP4 via ffmpeg, or GIF via Pillow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from . import style
from .render import density_colorbar, render_map


def first_urban_year(grid0, history) -> np.ndarray:
    """Year each cell first appears urban across the recorded frames."""
    years = np.full(grid0.shape, np.nan, dtype=np.float32)
    init = grid0["urban"] > 0
    years[init] = grid0["urbanized_at"][init]
    for f in history.frames:
        u = f.layers["urban"] > 0
        m = u & np.isnan(years)
        years[m] = f.year
    return years


def animate_history(grid0, history, out: str | Path, fps: int = 8, show_field: bool = False,
                    title: str | None = None, dpi: int = 110, grid_final=None,
                    quiet_kinds=("station",)) -> Path:
    """Render every recorded frame of ``history`` (started from ``grid0``).

    ``grid_final`` supplies the anchor list (scenario anchors are drawn with
    their status at each frame's year). Openings of ``quiet_kinds`` anchors are
    not marked on the timeline.
    """
    style.apply()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = grid_final or grid0
    first = first_urban_year(grid0, history)
    years = history.years
    has_density = "density" in history.frames[0].layers
    if has_density and np.isfinite(history.series("built_surface")).any():
        area = history.series("built_surface") * grid0.cell_size ** 2 / 1e6
        series_title = "Built-up area (km², Σ density × cell area)"
    else:
        area = history.series("urban_area")
        series_title = "Urban area (cells)"
    y0, y1 = history.frames[0].year, history.frames[-1].year

    fig = plt.figure(figsize=(7.2, 8.4))
    if has_density:
        ax_map = fig.add_axes([0.02, 0.24, 0.84, 0.72])
        cax = fig.add_axes([0.88, 0.36, 0.022, 0.48])
        density_colorbar(ax_map, cax=cax, label="Built-up density")
    else:
        ax_map = fig.add_axes([0.02, 0.24, 0.96, 0.72])
    ax_ts = fig.add_axes([0.11, 0.06, 0.84, 0.13])

    def draw(i):
        f = history.frames[i]
        ax_map.clear()
        ua = np.where(f.layers["urban"] > 0, first, np.nan)
        phi = f.layers.get("anchor_potential") if show_field else None
        render_map(ax_map, ref, urban=f.layers["urban"], roads=f.layers.get("roads", grid0["roads"]),
                   urbanized_at=ua, year=f.year, start_year=y0, end_year=y1, phi=phi,
                   density=f.layers.get("density"), colorbar=False,
                   title=f"{title or ref.name} — {f.year:g}")
        ax_ts.clear()
        ax_ts.plot(years, area, color=style.GRID, lw=2)
        k = years <= f.year
        ax_ts.plot(years[k], area[k], color=style.SERIES[0], lw=2)
        ax_ts.plot([f.year], [area[k][-1]], "o", ms=8, color=style.SERIES[0], mec=style.SURFACE, mew=2)
        ax_ts.set_title(series_title, loc="left", fontsize=9)
        ax_ts.set_xlim(years[0], years[-1])
        ax_ts.set_ylim(0, np.nanmax(area) * 1.08)
        for e in history.events:
            if e["year"] <= f.year and e.get("kind") not in quiet_kinds:
                ax_ts.axvline(e["year"], color=style.INK_MUTED, lw=1, ls="--")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(history.frames), blit=False)
    if out.suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out


def save_frames(grid0, history, out_dir: str | Path, every: int = 1, dpi: int = 110,
                grid_final=None) -> list[Path]:
    """Write individual PNG frames (handy for slides or custom compositing)."""
    style.apply()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = first_urban_year(grid0, history)
    ref = grid_final or grid0
    y0, y1 = history.frames[0].year, history.frames[-1].year
    paths = []
    for f in history.frames[::every]:
        fig, ax = plt.subplots(figsize=(7, 7))
        render_map(ax, ref, urban=f.layers["urban"], roads=f.layers.get("roads"),
                   urbanized_at=np.where(f.layers["urban"] > 0, first, np.nan),
                   year=f.year, start_year=y0, end_year=y1, density=f.layers.get("density"))
        p = out_dir / f"frame_{f.step:04d}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
