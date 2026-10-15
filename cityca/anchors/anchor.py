"""Anchor points: features that pull urban growth towards them.

An anchor is a point-like generator of attraction — a CBD, a station, a port,
a new airport. Each anchor has a *mass* that changes over time (announcement
-> construction -> opening -> maturity) and a *decay* kernel describing how its
influence falls off with distance. ``anchors.field`` sums them into a
potential field Φ that growth models can consume.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Default parameters per anchor kind. Anything given explicitly in a city or
# scenario file overrides these.
ANCHOR_KINDS: dict[str, dict[str, Any]] = {
    "generic": {},
    "cbd": {"mass": 1.0, "decay": {"type": "power", "scale": 8.0, "beta": 1.5}},
    "subcenter": {"mass": 0.5, "decay": {"type": "exponential", "scale": 10.0}},
    # Metro/rail stations: local (walking-catchment) pull that densifies and
    # extends the fringe, with little leapfrog seeding of its own.
    "station": {"mass": 0.25, "decay": {"type": "exponential", "scale": 4.0},
                "anticipation": 0.3, "ramp_years": 5.0, "ramp_shape": "saturating",
                "seeds": 0.05},
    # Office/IT parks: job magnets whose catchment fills in over a decade.
    "tech_park": {"mass": 0.5, "decay": {"type": "exponential", "scale": 12.0},
                  "anticipation": 0.2, "ramp_years": 10.0, "ramp_shape": "saturating",
                  "metric": "travel"},
    "university": {"mass": 0.4, "decay": {"type": "exponential", "scale": 8.0}},
    "port": {"mass": 0.6, "decay": {"type": "dog", "scale": 18.0, "inner": 3.0,
                                    "inner_strength": 0.6}},
    "industrial": {"mass": 0.5, "decay": {"type": "dog", "scale": 14.0, "inner": 2.5,
                                          "inner_strength": 0.5}},
    # Airports pull growth regionally but repel housing from the noise zone.
    # Zero crossing ≈ 8 cells: repulsive noise zone inside, attraction beyond.
    # ``capacity`` (cells) raises the carrying capacity of capacity-based demand laws.
    "airport": {"mass": 1.2, "decay": {"type": "dog", "scale": 25.0, "inner": 7.0,
                                       "inner_strength": 1.8},
                "footprint_radius": 4.0, "access_road": 3, "anticipation": 0.05,
                "ramp_years": 20.0, "ramp_shape": "logistic", "metric": "travel",
                "capacity": 1500},
}


@dataclass
class Anchor:
    name: str
    position: tuple[float, float]  # (x, y) in cell units: x = column, y = row
    kind: str = "generic"
    mass: float = 1.0
    decay: dict = field(default_factory=lambda: {"type": "exponential", "scale": 15.0})
    metric: str = "euclidean"  # "euclidean" or "travel" (road-network cost distance)
    announce_year: float | None = None  # when growth starts anticipating it
    open_year: float | None = None  # when it becomes operational (None = already open)
    anticipation: float = 0.25  # fraction of mass felt between announce and open
    ramp_years: float = 0.0  # years after opening to (nearly) reach full mass
    ramp_shape: str = "linear"  # linear | saturating | logistic (see ``maturity``)
    seeds: float = 1.0  # weight in leapfrog anchor seeding (0 = attraction only)
    footprint_radius: float = 0.0  # cells built/urbanised when it opens
    access_road: int = 0  # road class of a link to the network built on opening (0 = none)
    meta: dict = field(default_factory=dict)

    def maturity(self, years_open: float) -> float:
        """0→1 build-up after opening.

        * ``linear``: reaches 1 at ``ramp_years``.
        * ``saturating``: 1 - exp(-3t/T), fast at first (95 % at T).
        * ``logistic``: slow start, S-curve centred on T/2 (≈98 % at T) — an
          airport city takes years before development really gathers pace.
        """
        T = self.ramp_years
        if T <= 0 or years_open >= 50 * T:
            return 1.0
        t = max(years_open, 0.0)
        if self.ramp_shape == "linear":
            return min(1.0, t / T)
        if self.ramp_shape == "saturating":
            return 1.0 - math.exp(-3.0 * t / T)
        if self.ramp_shape == "logistic":
            k = 8.0 / T
            lo = 1.0 / (1.0 + math.exp(k * T / 2))
            return (1.0 / (1.0 + math.exp(-k * (t - T / 2))) - lo) / (1.0 - lo)
        raise ValueError(f"Unknown ramp_shape '{self.ramp_shape}'")

    def strength(self, year: float) -> float:
        """Effective mass at a given year (the temporal envelope)."""
        if self.open_year is None:
            return self.mass
        if year >= self.open_year:
            f = self.anticipation + (1 - self.anticipation) * self.maturity(year - self.open_year)
            return self.mass * f
        if self.announce_year is not None and year >= self.announce_year:
            span = max(self.open_year - self.announce_year, 1e-9)
            # Speculation builds up linearly until opening.
            return self.mass * self.anticipation * (year - self.announce_year) / span
        return 0.0

    def is_open(self, year: float) -> bool:
        return self.open_year is None or year >= self.open_year

    @property
    def rc(self) -> tuple[int, int]:
        return int(round(self.position[1])), int(round(self.position[0]))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position"] = list(self.position)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        d = dict(d)
        kind = d.get("kind", "generic")
        if kind not in ANCHOR_KINDS:
            raise KeyError(f"Unknown anchor kind '{kind}'. Known: {sorted(ANCHOR_KINDS)}")
        merged = {**ANCHOR_KINDS[kind], **d}
        merged["position"] = tuple(merged["position"])
        known = cls.__dataclass_fields__
        extra = {k: merged.pop(k) for k in list(merged) if k not in known}
        merged.setdefault("meta", {}).update(extra)
        return cls(**merged)
