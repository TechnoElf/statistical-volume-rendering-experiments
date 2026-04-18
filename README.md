# statistical-volume-rendering-experiments

## Setup

### Using Nix (recommended)

```bash
nix develop
```

This provides all dependencies including pyopenvdb, Vulkan drivers, and CUDA tooling.

### Using uv

```bash
uv sync
```

#### Additional dependencies

**pyopenvdb** is not available on PyPI and must be built from source. It requires
OpenVDB, Boost, TBB, and nanobind. See the
[OpenVDB build documentation](https://www.openvdb.org/documentation/doxygen/build.html)
for instructions on building the Python module.

**Vulkan** drivers and the Vulkan loader must be installed on your system for
slangpy to function. On most Linux distributions this is available via the
package manager (e.g. `vulkan-tools`, `libvulkan-dev`).

## Usage

```bash
volff gather            # Download volume datasets
volff prepare -s 24 -l 12  # Generate training data
volff train -e 8           # Train model
volff infer                # Run inference
```

All commands accept `--working-dir` / `-w` (default: `./run`).
