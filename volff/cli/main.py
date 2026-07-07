import math
import os
from pathlib import Path
from typing import Annotated
from urllib.request import urlretrieve

import numpy as np
import typer
from PIL import Image
from rich import print

from volff.cli import renderer
from volff.cli.renderer import RenderKind
from volff.constants import asset_sources
from volff.pipelines.relight import RelightPipeline
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
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    for name, info in asset_sources.items():
        print(f"[VLF] Retrieving {name}...")
        urlretrieve(info["url"], assets_dir / name)

    print("[VLF] Done.")


@cli.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    print("[VLF] Loading volume...")
    volume = load_vdb(assets_dir / "CT-Chest.vdb")

    print("[VLF] Rendering...")

    with RelightPipeline() as p:
        p.prepare(volume)
        p.train(
            {
                "iterations": 512,
                "pitch": math.pi / 2.0,
                "yaw": 0,
                "roll": math.pi / 2.0,
                "scale": 1.5,
                "ctx_path": "run/prompt_ctx_opt.pt",
            }
        )

    print("[VLF] Done.")


@cli.command()
def render(ctx: typer.Context, kind: RenderKind):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    print("[VLF] Loading volume...")
    volume = load_vdb(assets_dir / "CT-Chest.vdb")

    print("[VLF] Rendering...")
    img = renderer.render(
        kind,
        volume,
        {
            "iterations": 512,
            "pitch": math.pi / 2.0,
            "yaw": 0,
            "roll": math.pi / 2.0,
            "scale": 1.5,
            "ctx_path": "run/prompt_ctx_opt.pt",
        },
    )

    print("[VLF] Saving...")
    Image.fromarray((img * 255).astype(np.uint8)).save(
        config.working_dir / f"img_{kind.value}.png"
    )

    print("[VLF] Done.")


def main():
    cli()
