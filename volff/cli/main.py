import os
import subprocess
from pathlib import Path
from typing import Annotated
from urllib.request import urlretrieve

import numpy as np
import torch
import typer
from PIL import Image
from rich import print

from volff.constants import asset_sources
from volff.model import PathTracerModel
from volff.trace import Tracer
from volff.volume import load_vdb

cli = typer.Typer()


class Config:
    working_dir: Path

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir


@cli.callback()
def config(
    ctx: typer.Context,
    working_dir: Annotated[Path, typer.Option("--working-dir", "-w")] = Path("./run/"),
):
    ctx.obj = Config(working_dir=working_dir)


@cli.command()
def gather(ctx: typer.Context):
    config: Config = ctx.obj
    assets_dir: Path = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    for name, url in asset_sources.items():
        print(f"[VLF] Retrieving {name}...")
        urlretrieve(url, assets_dir / name)

    for name, url in asset_sources.items():
        print(f"[VLF] Processing {name}...")
        subprocess.run(["pvmdds", assets_dir / name])
        subprocess.run(["python", "scripts/pvm_to_vdb.py", assets_dir / name])

    print("[VLF] Done.")


@cli.command()
def prepare(ctx: typer.Context):
    config: Config = ctx.obj
    assets_dir: Path = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    model = PathTracerModel()
    model.load_state_dict(torch.load(config.working_dir / "model.pth"))

    with Tracer.create(1280, 720) as tracer:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "MRI-Head.vdb")

        print("[VLF] Pathtracing...")
        img = tracer.trace(volume, 1)

        print("[VLF] Inferring...")
        in_imgs = []
        for i in range(5):
            in_img = np.zeros((720, 1280, 4), dtype=np.float32)
            if i == 0:
                in_img[:, :, 0:3] = img
            in_img[:, :, 0:3] = (in_img[:, :, 0:3] * 2) - 1
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)
            in_imgs.append(in_img)

        in_imgs = torch.cat(in_imgs)

        out = model(in_imgs[None])
        img = out[0].permute(1, 2, 0).detach().numpy()

        print("[VLF] Saving...")
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(config.working_dir / "out.png")

    print("[VLF] Done.")


def main():
    cli()
