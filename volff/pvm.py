import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from volff.vdb import VDBGrid, write_vdb


def parse_pvm(path):
    with open(path, "rb") as f:
        data = f.read()

    [magic, data] = data.split(b"\n", 1)
    if magic != b"PVM3":
        print(
            f"error: {path} is not a valid PVM3 file (magic: {magic!r})",
            file=sys.stderr,
        )
        sys.exit(1)

    [size, data] = data.split(b"\n", 1)
    size = size.decode("utf-8").split(" ")
    if len(size) != 3:
        print(f"error: expected 3 dimensions, got {len(size)}", file=sys.stderr)
        sys.exit(1)
    [width, height, depth] = [int(x) for x in size]

    [voxel_size, data] = data.split(b"\n", 1)
    voxel_size = voxel_size.decode("utf-8").split(" ")
    if len(voxel_size) != 3:
        print(
            f"error: expected 3 voxel size components, got {len(voxel_size)}",
            file=sys.stderr,
        )
        sys.exit(1)
    [voxel_width, voxel_height, voxel_depth] = [float(x) for x in voxel_size]

    [components, data] = data.split(b"\n", 1)
    components = int(components.decode("utf-8"))

    print(f"Size: {width}x{height}x{depth}")
    print(f"Voxel Size: {voxel_width} {voxel_height} {voxel_depth}")
    print(f"Components: {components}")

    if components not in (1, 2):
        print(
            f"error: unsupported component count {components} (expected 1 or 2)",
            file=sys.stderr,
        )
        sys.exit(1)

    data_size = width * height * depth * components
    if len(data) < data_size:
        print(
            f"error: file truncated, expected {data_size} bytes of voxel data but got {len(data)}",
            file=sys.stderr,
        )
        sys.exit(1)

    metadata = data[data_size:]
    metadata = metadata.decode("utf-8").split("\0")
    data = data[:data_size]

    print("Metadata:")
    for line in metadata:
        print(line)

    volume = np.zeros((width, height, depth), dtype=np.float32)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if components == 1:
                    value = data[z * width * height + y * width + x]
                    volume[x, y, z] = float(value) / 255.0
                elif components == 2:
                    msb = data[(z * width * height + y * width + x) * 2 + 0]
                    lsb = data[(z * width * height + y * width + x) * 2 + 1]
                    value = (msb << 8) | lsb
                    volume[x, y, z] = float(value) / 32767.0

    return volume


def pvm_to_vdb(
    input: Path,
    output: Path | None = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
):
    if not input.exists():
        print(f"error: input file not found: {input}", file=sys.stderr)
        sys.exit(1)

    output = output if output is not None else input.with_suffix(".vdb")

    volume = parse_pvm(input)

    target = 256
    wx, wy, wz = volume.shape
    zoom_factors = (
        scale_x * target / wx,
        scale_y * target / wy,
        scale_z * target / wz,
    )
    volume = zoom(volume, zoom_factors, order=1).astype(np.float32)
    print(f"Resampled volume to {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}")

    def center_slice(size):
        if size >= target:
            start = (size - target) // 2
            return slice(start, start + target), slice(None)
        else:
            start = (target - size) // 2
            return slice(None), slice(start, start + size)

    sx, sy, sz = volume.shape
    xs, xd = center_slice(sx)
    ys, yd = center_slice(sy)
    zs, zd = center_slice(sz)

    result = np.zeros((target, target, target), dtype=np.float32)
    result[xd, yd, zd] = volume[xs, ys, zs]
    volume = result
    print(f"Output volume: {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}")

    grid = VDBGrid("density")
    grid.set_dense((0, 0, 0), volume)
    print(f"Active voxels: {grid.active_count()}")
    write_vdb(str(output), grid)
    print(f"Wrote {output}")
