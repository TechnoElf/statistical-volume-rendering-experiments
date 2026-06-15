from pathlib import Path

import numpy as np

from volff.vdb import read_vdb


def load_vdb(path: Path):
    grid = read_vdb(str(path))
    print(f"Active voxels: {grid.active_count()}")
    volume = grid.to_dense((0, 0, 0), (256, 256, 256))
    volume = volume / np.max(volume)
    return volume
