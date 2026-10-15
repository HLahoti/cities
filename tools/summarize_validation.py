"""Summarise a rolling-origin validation.json into per-method tables.

    python3 -P -E tools/summarize_validation.py runs/<...>_validate [--exclude-target 2019]
"""

import json
import sys

import numpy as np


def main(path, exclude=()):
    rep = json.load(open(f"{path}/validation.json"))
    r = [x for x in rep["results"] if x["target"] not in exclude]
    methods = list(dict.fromkeys(x["method"] for x in r))
    hs = sorted({x["horizon"] for x in r})
    short = [h for h in hs if h <= 4]
    print(f"{'method':24s} {'FoM all':>8s} {'FoM ≤4y':>8s} {'|area err|':>10s} {'area bias':>9s} "
          f"{'dens skill':>10s} {'dens all':>9s}  n")
    for m in methods:
        rows = [x for x in r if x["method"] == m]

        def mean(k, rows=rows):
            v = [x[k] for x in rows if x.get(k) is not None]
            return np.mean(v) if v else float("nan")
        fs = np.mean([x["fom"] for x in rows if x["horizon"] in short]) if short else float("nan")
        print(f"{m:24s} {mean('fom'):8.3f} {fs:8.3f} {100 * np.mean([abs(x['area_err']) for x in rows]):9.1f}% "
              f"{100 * mean('area_err'):+8.1f}% {mean('density_skill'):+10.3f} {mean('density_skill_all'):+9.3f}  {len(rows)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    excl = tuple(float(a) for a in args[args.index("--exclude-target") + 1:]) if "--exclude-target" in args else ()
    main(args[0], excl)
