# AGENTS.md

## Project Overview

Statistical volume rendering experiments using neural networks. A UNet-based model learns to denoise/enhance pathtraced volume renderings by combining low-sample pathtraced images with isosurface renders at multiple thresholds.

## Tech Stack

- **Language:** Python 3.13
- **ML Framework:** PyTorch
- **GPU Compute:** Vulkan via slangpy, with Slang shaders (`volff/kernels/`)
- **Volume Format:** OpenVDB
- **CLI:** Typer + Rich
- **Environment:** Nix Flakes (`flake.nix`)

## Project Structure

```
volff/                  # Main package
├── cli/main.py         # CLI commands: gather, trace, prepare, train, infer
├── kernels/            # GPU shaders (Slang)
│   ├── common.slang    # RNG, ray-box intersection, phase functions
│   ├── pathtracer.slang
│   └── raycaster.slang
├── model.py            # UNet architecture (PathTracerModel)
├── dataset.py          # PathTracerDataset, tiling utilities
├── trace.py            # Vulkan/slangpy GPU rendering
├── transform.py        # Camera and projection matrices
├── volume.py           # OpenVDB volume loading
└── constants.py        # Asset URLs
scripts/                # Conversion utilities
run/                    # Working directory (assets, dataset, model weights)
```

## Setup & Running

```bash
nix develop             # Enter dev environment with all dependencies
python -m volff gather  # Download volume datasets
python -m volff prepare -s 24 -l 12  # Generate training data
python -m volff train -e 8           # Train model
python -m volff infer                # Run inference
```

All commands accept `--working-dir` / `-w` (default: `./run`).

## Code Conventions

- GPU shaders are written in Slang, not GLSL/HLSL
- The model input is 20 channels (5 images x 4 RGBA channels)
- Images are tiled into 512x512 patches with overlap blending for training and inference
- Dependencies are managed entirely through `flake.nix`, not pip/pyproject.toml
