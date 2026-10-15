"""Growth-model interface.

A growth model looks at the grid and converts some non-urban cells to urban
in one time step. Everything else — time keeping, anchors, events, recording
— is handled by ``Simulation`` so models stay small and swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..registry import Registry

MODELS = Registry("growth model")


@dataclass
class StepContext:
    step: int
    year: float
    dt: float
    rng: np.random.Generator
    info: dict = field(default_factory=dict)


class GrowthModel(ABC):
    name = "base"

    def reset(self, grid) -> None:
        """Called once before the first step (and between Monte Carlo runs)."""

    @abstractmethod
    def step(self, grid, ctx: StepContext) -> dict:
        """Advance one step; returns stats (must include ``"new"``: cells added)."""

    def params(self) -> dict:
        """Current (possibly self-modified) parameters, for logging."""
        return {}

    def set_param(self, key: str, value) -> None:
        if not hasattr(self, key):
            raise AttributeError(f"{self.name} has no parameter '{key}'")
        setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params()})"
