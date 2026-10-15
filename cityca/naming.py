"""Output directory naming: ``<city_name>_<datetime>_<run_type>``.

e.g. ``runs/riverton_20261015-143502_airport``. The timestamp is local time,
filesystem-safe (no colons) and sorts chronologically.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def slug(text: str) -> str:
    """Lowercase, keep [a-z0-9-], collapse everything else to '-'."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "unnamed"


def run_dir(city_name: str, run_type: str, root: str | Path = "runs",
            when: datetime | None = None, create: bool = True) -> Path:
    """Fresh output directory under ``root``; never reuses an existing one."""
    stamp = (when or datetime.now()).strftime(TIMESTAMP_FORMAT)
    base = Path(root) / f"{slug(city_name)}_{stamp}_{slug(run_type)}"
    path, n = base, 2
    while path.exists():  # two runs in the same second
        path = base.with_name(f"{base.name}-{n}")
        n += 1
    if create:
        path.mkdir(parents=True)
    return path
