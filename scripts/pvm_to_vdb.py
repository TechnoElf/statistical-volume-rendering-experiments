import argparse
import sys
from pathlib import Path

import numpy as np
import openvdb as vdb


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PVM3 volume files to OpenVDB format."
    )
    parser.add_argument("input", type=Path, help="path to the input .pvm file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="path to the output .vdb file (default: input with .vdb extension)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output if args.output else args.input.with_suffix(".vdb")

    volume = parse_pvm(args.input)

    grid = vdb.FloatGrid()
    grid.copyFromArray(volume)
    vdb.write(str(output), grid)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
