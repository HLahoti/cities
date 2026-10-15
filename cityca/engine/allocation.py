"""Allocators: turn a transition-potential field into cells to convert."""

from __future__ import annotations

import numpy as np

from ..registry import Registry

ALLOCATORS = Registry("allocator")


@ALLOCATORS.register("top_n")
class TopN:
    """Deterministic: the n highest-potential cells (White–Engelen style)."""

    def __call__(self, potential: np.ndarray, n: int | None, rng) -> np.ndarray:
        mask = np.zeros(potential.shape, dtype=bool)
        flat = potential.ravel()
        n = min(int(n or 0), int((flat > 0).sum()))
        if n > 0:
            idx = np.argpartition(flat, -n)[-n:]
            mask.ravel()[idx] = True
        return mask


@ALLOCATORS.register("roulette")
class Roulette:
    """Sample n cells without replacement with probability ∝ P^gamma.

    gamma → ∞ approaches top_n; gamma = 1 is proportional; small gamma is
    more scattered (more 'organic').
    """

    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma

    def __call__(self, potential, n, rng):
        mask = np.zeros(potential.shape, dtype=bool)
        w = np.maximum(potential.ravel(), 0.0)
        if self.gamma != 1:
            m = w.max()
            w = (w / m) ** self.gamma if m > 0 else w
        nz = np.flatnonzero(w > 0)
        n = min(int(n or 0), nz.size)
        if n > 0:
            p = w[nz] / w[nz].sum()
            idx = rng.choice(nz, size=n, replace=False, p=p)
            mask.ravel()[idx] = True
        return mask


@ALLOCATORS.register("stochastic")
class Stochastic:
    """Unconstrained: each cell converts with probability min(1, scale·P).

    Ignores demand; total growth emerges from the rules (classic CA).
    """

    def __init__(self, scale: float = 0.1, normalize: bool = True):
        self.scale, self.normalize = scale, normalize

    def __call__(self, potential, n, rng):
        p = potential
        if self.normalize and p.max() > 0:
            p = p / p.max()
        return rng.random(p.shape) < np.clip(self.scale * p, 0.0, 1.0)
