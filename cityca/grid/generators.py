"""Procedural layer generators (terrain etc.), usable from city files:

    "elevation": {"source": "generator", "generator": "fractal_noise",
                  "params": {"amplitude": 120, "beta": 3.0, "seed": 4}}
"""

from __future__ import annotations

import numpy as np

from ..registry import Registry

GENERATORS = Registry("layer generator")


@GENERATORS.register("fractal_noise")
def fractal_noise(shape, beta: float = 3.0, amplitude: float = 100.0,
                  offset: float = 0.0, seed: int = 0, **_):
    """Spectral-synthesis fractional Brownian surface (power spectrum ~ f^-beta)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, amp.shape)
    z = np.fft.irfft2(amp * np.exp(1j * phase), s=shape)
    z = (z - z.min()) / (np.ptp(z) or 1.0)
    return offset + amplitude * z


@GENERATORS.register("gaussian_hills")
def gaussian_hills(shape, hills=(), base: float = 0.0, **_):
    """Sum of Gaussian bumps: hills = [{"x":, "y":, "height":, "radius":}, ...]."""
    rr, cc = np.indices(shape)
    z = np.full(shape, float(base))
    for h in hills:
        z += h["height"] * np.exp(-((cc - h["x"]) ** 2 + (rr - h["y"]) ** 2) / (2 * h["radius"] ** 2))
    return z


@GENERATORS.register("plane")
def plane(shape, value: float = 0.0, dx: float = 0.0, dy: float = 0.0, **_):
    """value + dx*col + dy*row."""
    rr, cc = np.indices(shape)
    return value + dx * cc + dy * rr


@GENERATORS.register("random")
def random_field(shape, low: float = 0.0, high: float = 1.0, seed: int = 0, **_):
    return np.random.default_rng(seed).uniform(low, high, shape)
