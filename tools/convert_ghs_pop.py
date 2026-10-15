"""Crop GHS-POP 30ss (1 km) tiles to the Bangalore box and save as .npy.

GHS-POP tiles are LZW-compressed BigTIFFs, which need `imagecodecs`. This
script runs in an isolated environment so the main package doesn't depend on it:

    uv run --no-project --with tifffile --with imagecodecs --with numpy \
        python -I tools/convert_ghs_pop.py data/raw/ghsl data/derived/ghs_pop
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

SOUTH, NORTH, WEST, EAST = 12.70, 13.35, 77.30, 77.95  # a little wider than the model grid


def main(raw: str, out: str):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tif in sorted(Path(raw).glob("GHS_POP_E*_30ss_*/*.tif")):
        with tifffile.TiffFile(tif) as t:
            pg = t.pages[0]
            sx, sy, _ = pg.tags["ModelPixelScaleTag"].value
            _, _, _, lon0, lat0, _ = pg.tags["ModelTiepointTag"].value
            a = pg.asarray()
        c0, c1 = int((WEST - lon0) / sx), int(np.ceil((EAST - lon0) / sx))
        r0, r1 = int((lat0 - NORTH) / sy), int(np.ceil((lat0 - SOUTH) / sy))
        crop = a[r0:r1, c0:c1].astype(np.float64)
        crop[crop < 0] = 0
        epoch = tif.name.split("_E")[1][:4]
        np.save(out_dir / f"pop_{epoch}.npy", crop)
        np.save(out_dir / "pop_georef.npy", np.array([lon0 + c0 * sx, lat0 - r0 * sy, sx, sy]))
        print(epoch, crop.shape, f"{crop.sum():,.0f} people in box")


if __name__ == "__main__":
    main(*sys.argv[1:3])
