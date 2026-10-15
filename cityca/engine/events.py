"""Scheduled interventions applied during a simulation.

Each event fires once, at the first step whose year is >= ``year``:

    {"type": "add_road", "year": 1985, "coords": [[10, 50], [120, 60]], "value": 3}
    {"type": "add_anchor", "year": 1990, "anchor": {"name": "Airport", "kind": "airport", ...}}
    {"type": "burn", "year": 2000, "layer": "excluded", "value": 1,
     "geometry": "polygon", "coords": [...]}            # e.g. a new greenbelt
    {"type": "urbanize", "year": 1995, "geometry": "circle", "center": [80, 80], "radius": 5}
    {"type": "set_param", "year": 2000, "param": "spread", "value": 10}
"""

from __future__ import annotations

import numpy as np

from ..grid.raster import burn, feature_mask
from ..registry import Registry

EVENTS = Registry("event")


class Event:
    name = "event"

    def __init__(self, year: float, label: str | None = None):
        self.year = float(year)
        self.label = label or self.name
        self.fired = False

    def apply(self, sim) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(year={self.year:g}, {self.label})"


@EVENTS.register("add_road")
class AddRoad(Event):
    """Build (or upgrade, since burning uses max) roads: one ``coords`` polyline
    or several in ``lines``."""
    name = "add_road"

    def __init__(self, year, coords=None, lines=None, value: int = 2, width: float = 1, **kw):
        super().__init__(year, **kw)
        self.lines = list(lines or []) + ([coords] if coords is not None else [])
        self.value, self.width = value, width

    def apply(self, sim):
        g = sim.grid
        m = np.zeros(g.shape, dtype=bool)
        for line in self.lines:
            m |= feature_mask(g.shape, {"geometry": "polyline", "coords": line, "width": self.width})
        burn(g["roads"], m, self.value, "max")
        g.touch("roads")
        return {"cells": int(m.sum())}


@EVENTS.register("add_anchor")
class AddAnchor(Event):
    name = "add_anchor"

    def __init__(self, year, anchor: dict, **kw):
        kw.setdefault("label", anchor.get("name", "anchor"))
        super().__init__(year, **kw)
        self.anchor = anchor

    def apply(self, sim):
        sim.grid.add_anchor(dict(self.anchor))
        return {"anchor": self.anchor.get("name")}


@EVENTS.register("remove_anchor")
class RemoveAnchor(Event):
    name = "remove_anchor"

    def __init__(self, year, anchor: str, **kw):
        super().__init__(year, **kw)
        self.anchor_name = anchor

    def apply(self, sim):
        sim.grid.remove_anchor(self.anchor_name)
        return {}


@EVENTS.register("burn")
class Burn(Event):
    """Burn a geometry into any layer (zoning, protection, new water, ...)."""
    name = "burn"

    def __init__(self, year, layer: str, value: float = 1, op: str = "set", **geom):
        label = geom.pop("label", None)
        super().__init__(year, label=label or f"burn:{layer}")
        self.layer, self.value, self.op, self.geom = layer, value, op, geom

    def apply(self, sim):
        g = sim.grid
        m = feature_mask(g.shape, self.geom)
        burn(g.ensure(self.layer), m, self.value, self.op)
        g.touch(self.layer)
        return {"cells": int(m.sum())}


@EVENTS.register("urbanize")
class Urbanize(Event):
    """Planned development: force a geometry to urban."""
    name = "urbanize"

    def __init__(self, year, **geom):
        label = geom.pop("label", None)
        super().__init__(year, label=label)
        self.geom = geom

    def apply(self, sim):
        g = sim.grid
        m = feature_mask(g.shape, self.geom) & ~g["blocked"]
        return {"cells": g.urbanize(m)}


@EVENTS.register("set_param")
class SetParam(Event):
    """Change a model parameter (policy shift, economic shock)."""
    name = "set_param"

    def __init__(self, year, param: str, value, target: str = "model", **kw):
        super().__init__(year, **kw)
        self.param, self.value, self.target = param, value, target

    def apply(self, sim):
        obj = sim.model
        for part in self.target.split(".")[1:]:  # e.g. "model.demand"
            obj = getattr(obj, part)
        if hasattr(obj, "set_param"):
            obj.set_param(self.param, self.value)
        else:
            setattr(obj, self.param, self.value)
        return {self.param: self.value}


def build_events(specs) -> list[Event]:
    return [EVENTS.build(s) for s in (specs or [])]
