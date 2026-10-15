"""Generic plugin registry.

Every swappable piece of the system (layer sources, derived properties, growth
models, factors, kernels, demand laws, allocators, events, metrics) lives in a
``Registry``. Components register themselves with a decorator and are built
from plain dict specs, so a JSON scenario can name any combination of them:

    {"type": "neighborhood", "radius": 4, "weight": 1.0}
"""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def deco(obj):
            key = name or getattr(obj, "name", None) or obj.__name__
            if key in self._items:
                raise KeyError(f"{self.kind} '{key}' registered twice")
            self._items[key] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            ) from None

    def build(self, spec: str | dict | Any, **extra) -> Any:
        """Instantiate from ``"name"`` or ``{"type": "name", **params}``.

        Objects that are already built (not str/dict) are passed through, so
        Python callers can mix specs and instances freely.
        """
        if isinstance(spec, str):
            return self.get(spec)(**extra)
        if isinstance(spec, dict):
            params = dict(spec)
            try:
                name = params.pop("type")
            except KeyError:
                raise ValueError(f"{self.kind} spec needs a 'type' key: {spec}") from None
            return self.get(name)(**params, **extra)
        return spec

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"
