"""Distance-decay functions and 2-D neighbourhood kernels.

Decay functions map a distance array ``d`` (in cells) to an influence weight.
They are shared by neighbourhood effects (how much nearby development
attracts more development) and by anchor potential fields (how far the pull
of an airport or CBD reaches).

All decays are normalised so that ``f(0) == 1`` (except ``dog``, which is a
signed "Mexican hat": attractive at mid range, repulsive close in).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .registry import Registry

DECAY = Registry("decay function")


@DECAY.register("exponential")
def exponential(d, scale: float = 10.0, **_):
    """exp(-d/scale): the classic gravity/potential-model decay."""
    return np.exp(-np.asarray(d, dtype=float) / scale)


@DECAY.register("gaussian")
def gaussian(d, scale: float = 10.0, **_):
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * (d / scale) ** 2)


@DECAY.register("power")
def power(d, scale: float = 1.0, beta: float = 2.0, **_):
    """(1 + d/scale)^-beta: heavy-tailed, scale-free beyond ``scale``."""
    return (1.0 + np.asarray(d, dtype=float) / scale) ** (-beta)


@DECAY.register("linear")
def linear(d, scale: float = 10.0, **_):
    """Linear fall-off reaching zero at d = scale."""
    return np.clip(1.0 - np.asarray(d, dtype=float) / scale, 0.0, 1.0)


@DECAY.register("step")
def step(d, scale: float = 10.0, **_):
    return (np.asarray(d, dtype=float) <= scale).astype(float)


@DECAY.register("dog")
def difference_of_gaussians(d, scale: float = 10.0, inner: float = 2.0,
                            inner_strength: float = 1.5, **_):
    """Mexican-hat kernel: broad attraction minus a narrow repulsive core.

    Useful for nuisance anchors (airports, ports, industry) that pull jobs and
    growth into the region but push housing away from the immediate vicinity.
    """
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * (d / scale) ** 2) - inner_strength * np.exp(-0.5 * (d / inner) ** 2)


def decay(spec: str | dict, d) -> np.ndarray:
    """Evaluate a decay given as ``"name"`` or ``{"type": name, **params}``."""
    if isinstance(spec, str):
        return DECAY.get(spec)(d)
    params = dict(spec)
    return DECAY.get(params.pop("type"))(d, **params)


# --- 2-D neighbourhood kernels -------------------------------------------------

def moore(radius: int = 1) -> np.ndarray:
    k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=float)
    k[radius, radius] = 0
    return k


def von_neumann(radius: int = 1) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = (np.abs(x) + np.abs(y) <= radius).astype(float)
    k[radius, radius] = 0
    return k


def disk(radius: float, decay_spec: str | dict | None = None,
         include_center: bool = False) -> np.ndarray:
    """Circular kernel of given radius, optionally weighted by a decay."""
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(x, y)
    k = (d <= radius).astype(float)
    if decay_spec is not None:
        k = k * decay(decay_spec, d)
    if not include_center:
        k[r, r] = 0
    return k


def convolve(arr: np.ndarray, kernel: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Zero-padded 2-D convolution; uses FFT for large kernels."""
    a = np.asarray(arr, dtype=float)
    if kernel.size > 49:
        out = signal.oaconvolve(a, kernel, mode="same")
        out[np.abs(out) < 1e-9] = 0.0  # FFT round-off
    else:
        out = ndimage.convolve(a, kernel, mode="constant", cval=0.0)
    if normalize:
        s = kernel.sum()
        if s:
            out /= s
    return out


# Offsets of the 8 Moore neighbours, used by rule-based models.
MOORE_OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1),
                          (0, 1), (1, -1), (1, 0), (1, 1)], dtype=int)
