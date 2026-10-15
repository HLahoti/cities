"""Recording simulation output: layer snapshots + per-step records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..metrics.morphology import DEFAULT_METRICS


@dataclass
class RecordSpec:
    every: int = 1  # snapshot layers every N steps (records are kept every step)
    layers: list[str] = field(default_factory=lambda: ["urban", "roads", "anchor_potential"])
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))

    @classmethod
    def build(cls, spec) -> "RecordSpec":
        if spec is None:
            return cls()
        if isinstance(spec, cls):
            return spec
        return cls(**spec)


@dataclass
class Frame:
    step: int
    year: float
    layers: dict[str, np.ndarray]


class History:
    def __init__(self, meta: dict | None = None):
        self.frames: list[Frame] = []
        self.records: list[dict] = []
        self.events: list[dict] = []
        self.meta = dict(meta or {})

    # --- access -------------------------------------------------------------------
    @property
    def years(self) -> np.ndarray:
        return np.array([r["year"] for r in self.records])

    def series(self, key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in self.records], dtype=float)

    def stack(self, layer: str) -> np.ndarray:
        return np.stack([f.layers[layer] for f in self.frames])

    def frame_at(self, year: float) -> Frame:
        return min(self.frames, key=lambda f: abs(f.year - year))

    @property
    def final(self) -> Frame:
        return self.frames[-1]

    def __len__(self):
        return len(self.frames)

    # --- persistence ----------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        names = sorted({k for f in self.frames for k in f.layers})
        arrays = {"steps": np.array([f.step for f in self.frames]),
                  "years": np.array([f.year for f in self.frames])}
        for n in names:
            arrays[f"layer_{n}"] = np.stack([f.layers[n] for f in self.frames])
        np.savez_compressed(d / "history.npz", **arrays)
        (d / "records.json").write_text(json.dumps(
            {"meta": self.meta, "records": self.records, "events": self.events},
            indent=1, default=_jd))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "History":
        d = Path(directory)
        doc = json.loads((d / "records.json").read_text())
        h = cls(doc.get("meta"))
        h.records, h.events = doc["records"], doc.get("events", [])
        z = np.load(d / "history.npz")
        names = [k[6:] for k in z.files if k.startswith("layer_")]
        for i, (s, y) in enumerate(zip(z["steps"], z["years"])):
            h.frames.append(Frame(int(s), float(y), {n: z[f"layer_{n}"][i] for n in names}))
        return h


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
