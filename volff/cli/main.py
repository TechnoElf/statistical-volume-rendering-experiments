import math
import os
from pathlib import Path
from typing import Annotated
from urllib.request import urlretrieve

import numpy as np
import torch
import typer
from PIL import Image
from rich import print
from torch import nn
from torch.utils.data import DataLoader

from volff.constants import asset_sources
from volff.dataset import PathTracerDataset, random_sample
from volff.hfen import HFENL1Loss
from volff.models.denoise import SimplePathTracerDenoiseModel
from volff.pipelines.denoise import DenoisePipeline
from volff.pipelines.pathtrace import PathTracePipeline
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


@cli.command()
def trace(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    with PathTracePipeline() as p:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "CT-Chest.vdb")
        p.prepare(volume)

        print(f"[VLF] Pathtracing...")
        img = p.render(
            {
                "iterations": 512,
                "pitch": math.pi / 2.0,
                "yaw": 0,
                "roll": math.pi / 2.0,
            }
        )

        print("[VLF] Saving...")
        Image.fromarray((img * 255).astype(np.uint8)).save(
            config.working_dir / f"img_ref.png"
        )


@cli.command()
def infer(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    with DenoisePipeline() as p:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "CT-Chest.vdb")
        p.prepare(volume)

        print("[VLF] Rendering...")
        img = p.render(
            {"model_path": config.working_dir / "model.pth", "yaw": math.pi / 2.0}
        )

        print("[VLF] Saving...")
        Image.fromarray((img * 255).astype(np.uint8)).save(
            config.working_dir / "img.png"
        )

        print("[VLF] Done.")


@cli.command()
def prepare(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option("--samples", "-s")] = 24,
    validation_samples: Annotated[int, typer.Option("--validation", "-l")] = 12,
):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)
    dataset_dir = config.working_dir / "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    assets = [
        (assets_dir / asset).absolute()
        for asset in os.listdir(assets_dir)
        if asset.endswith(".vdb")
    ]

    samples = int(samples / len(assets))
    validation_samples = int(validation_samples / len(assets))

    with Tracer.create(1280, 720) as tracer:
        for i, asset in enumerate(assets):
            print(f"[VLF] Loading volume {i + 1}/{len(assets)}...")
            volume = load_vdb(asset)

            for j in range(int(samples / 4)):
                index = samples * i + j * 4
                print(f"[VLF] Sample {index + 1}/{samples * len(assets)}...")
                random_sample(index, False, dataset_dir, volume, tracer, 4)

            for j in range(int(validation_samples / 4)):
                index = validation_samples * i + j * 4
                print(
                    f"[VLF] Validation {index + 1}/{validation_samples * len(assets)}..."
                )
                random_sample(index, True, dataset_dir, volume, tracer, 4)

    print("[VLF] Done.")


@cli.command()
def train(
    ctx: typer.Context,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 8,
):
    config = ctx.obj
    dataset_dir = config.working_dir / "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VLF] Using device: {device}")

    model = SimplePathTracerDenoiseModel()
    if os.path.exists(config.working_dir / "model.pth"):
        print(f"[VLF] Loading existing model")
        s = torch.load(config.working_dir / "model.pth")
        model.load_state_dict(s)
    else:
        print(f"[VLF] Initializing new model")

    model.to(device)

    l1_loss_fn = nn.L1Loss()
    hfen_loss_fn = HFENL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.1
    )

    train_dataset = PathTracerDataset(dataset_dir, train=True)
    val_dataset = PathTracerDataset(dataset_dir, train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            o = outputs.clamp(min=1e-6).pow(0.2)  # .sign() * outputs.abs().pow(0.2)
            t = targets.clamp(min=1e-6).pow(0.2)  # .sign() * targets.abs().pow(0.2)
            loss = 0.8 * l1_loss_fn(o, t) + 0.1 * hfen_loss_fn(o, t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                o = outputs.clamp(min=1e-6).pow(0.2)  # .sign() * outputs.abs().pow(0.2)
                t = targets.clamp(min=1e-6).pow(0.2)  # .sign() * targets.abs().pow(0.2)
                loss = 0.8 * l1_loss_fn(o, t) + 0.1 * hfen_loss_fn(o, t)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"[VLF] Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"LR: {scheduler._last_lr[0]:.8f}"
        )

        torch.save(model.state_dict(), config.working_dir / "model.pth")

    print("[VLF] Done.")


@cli.command()
def generate(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    with RelightPipeline() as p:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "CT-Chest.vdb")
        p.prepare(volume)

        print("[VLF] Rendering...")
        # p.train(
        #     {
        #         "pitch": math.pi / 2.0,
        #         "yaw": 0,
        #         "roll": math.pi / 2.0,
        #     }
        # )

        img = p.render(
            {
                "pitch": math.pi / 2.0,
                "yaw": 0,
                "roll": math.pi / 2.0,
            }
        )

        print("[VLF] Saving...")
        Image.fromarray((img * 255).astype(np.uint8)).save(
            config.working_dir / "img_flux.png"
        )

    print("[VLF] Done.")


def main():
    cli()
