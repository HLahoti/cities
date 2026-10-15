"""The simulation loop.

Per step (year t → t + dt):

1. advance the clock;
2. fire scheduled events whose year has arrived;
3. open anchors whose ``open_year`` has arrived (footprint + access road);
4. recompute the anchor potential field Φ(t) into ``grid["anchor_potential"]``;
5. run the growth model, then any extra processes (e.g. densification);
6. record stats, metrics and (every N steps) layer snapshots.
"""

from __future__ import annotations

import numpy as np

from ..anchors import effects
from ..anchors.field import AnchorField
from ..metrics.morphology import compute as compute_metrics
from .base import GrowthModel, StepContext
from .events import Event
from .history import Frame, History, RecordSpec


class Simulation:
    def __init__(self, grid, model: GrowthModel, field: AnchorField | None = None,
                 events: list[Event] | None = None, dt: float = 1.0,
                 seed: int | None = None, record: RecordSpec | dict | None = None,
                 processes: list | None = None):
        self.grid = grid
        self.model = model
        self.processes = list(processes or [])
        if "urban_lag" in grid.layers and not any(p.name == "track_lags" for p in self.processes):
            from .processes import TrackLags
            self.processes.append(TrackLags())  # keep momentum layers current
        self.field = field or AnchorField()
        # Events dated at or before the start year are history: already in the grid.
        self.events = sorted((e for e in (events or []) if e.year > grid.year), key=lambda e: e.year)
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.record = RecordSpec.build(record)
        self.step_index = 0
        self.history = History(meta={"city": grid.name, "model": type(model).__name__,
                                     "seed": seed, "start_year": grid.year})
        self._opened = {a.name for a in grid.anchors
                        if a.open_year is None or a.open_year < grid.year}
        self.model.reset(grid)
        for p in self.processes:
            p.reset(grid)
        self._update_field()
        self._record({"new": 0}, snapshot=True)

    # --- internals -------------------------------------------------------------
    def _update_field(self) -> None:
        if self.grid.anchors:
            self.grid["anchor_potential"] = self.field.compute(self.grid)
        elif "anchor_potential" not in self.grid:
            self.grid.ensure("anchor_potential")

    def _fire_events(self) -> None:
        for ev in self.events:
            if not ev.fired and self.grid.year >= ev.year:
                info = ev.apply(self)
                ev.fired = True
                self.history.events.append({"year": self.grid.year, "event": ev.label,
                                            "type": ev.name, **(info or {})})

    def _open_anchors(self) -> None:
        for a in self.grid.anchors:
            if a.name not in self._opened and a.is_open(self.grid.year):
                info = effects.activate(self.grid, a)
                self._opened.add(a.name)
                self.history.events.append({"year": self.grid.year, "event": f"open:{a.name}",
                                            "type": "anchor_open", "kind": a.kind, **info})

    def _record(self, stats: dict, snapshot: bool) -> None:
        rec = {"step": self.step_index, "year": self.grid.year, **stats}
        rec.update(compute_metrics(self.grid, self.record.metrics))
        self.history.records.append(rec)
        if snapshot:
            layers = {n: self.grid[n].copy() for n in self.record.layers if self.grid.has(n)}
            self.history.frames.append(Frame(self.step_index, self.grid.year, layers))

    # --- public ------------------------------------------------------------------
    def step(self) -> dict:
        self.step_index += 1
        self.grid.set_year(self.grid.year + self.dt)
        self._fire_events()
        self._open_anchors()
        self._update_field()
        ctx = StepContext(self.step_index, self.grid.year, self.dt, self.rng)
        stats = self.model.step(self.grid, ctx)
        for p in self.processes:
            stats.update(p.step(self.grid, ctx) or {})
        snap = self.step_index % max(self.record.every, 1) == 0
        self._record(stats, snap)
        return stats

    def run(self, steps: int | None = None, until: float | None = None,
            progress: bool = False) -> History:
        if steps is None:
            if until is None:
                raise ValueError("give steps or until")
            steps = int(np.ceil((until - self.grid.year) / self.dt - 1e-9))
        for i in range(steps):
            stats = self.step()
            if progress:
                print(f"\r  year {self.grid.year:7.1f}  +{stats.get('new', 0):5d} cells  "
                      f"urban {int(self.grid['urban'].sum()):7d}", end="", flush=True)
        if progress:
            print()
        # Always keep the final state.
        if self.history.frames[-1].step != self.step_index:
            self._record_final_snapshot()
        return self.history

    def _record_final_snapshot(self) -> None:
        layers = {n: self.grid[n].copy() for n in self.record.layers if self.grid.has(n)}
        self.history.frames.append(Frame(self.step_index, self.grid.year, layers))
