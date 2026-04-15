from pathlib import Path

import numpy as np
import openvdb as vdb


def load_vdb(path: Path):
    grid, metadata = vdb.readAll(str(path))
    grid = grid[0]
    acc = grid.getConstAccessor()
    start = grid.metadata["file_bbox_min"]
    end = grid.metadata["file_bbox_max"]
    size = (end[0] - start[0] + 1, end[1] - start[1] + 1, end[2] - start[2] + 1)

    volume = np.zeros(size, dtype=np.float32)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                density, active = acc.probeValue(
                    (
                        start[0] + x,
                        start[1] + y,
                        start[2] + z,
                    )
                )
                volume[x, y, z] = density

    return volume
