import math
import os
import random
import subprocess
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
from volff.dataset import PathTracerDataset, create_sample
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
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
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
def trace(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    with Tracer.create(1280, 720) as tracer:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "MRI-Head.vdb")

        print("[VLF] Pathtracing...")
        img = tracer.trace(volume, 256, yaw=math.pi / 2.0)

        print("[VLF] Saving...")
        Image.fromarray((img * 255).astype(np.uint8)).save(
            config.working_dir / "img_ref.png"
        )


@cli.command()
def infer(ctx: typer.Context):
    config = ctx.obj
    assets_dir = config.working_dir / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    model = PathTracerModel()
    model.load_state_dict(torch.load(config.working_dir / "model.pth"))

    with Tracer.create(1280, 720) as tracer:
        print("[VLF] Loading volume...")
        volume = load_vdb(assets_dir / "MRI-Head.vdb")

        print("[VLF] Pathtracing...")
        img_pt = tracer.trace(volume, 1, yaw=math.pi / 2.0)

        print("[VLF] Raycasting...")
        img_iso_1 = tracer.isosurface(volume, 0.10, yaw=math.pi / 2.0)
        img_iso_2 = tracer.isosurface(volume, 0.25, yaw=math.pi / 2.0)
        img_iso_6 = tracer.isosurface(volume, 0.65, yaw=math.pi / 2.0)
        img_iso_9 = tracer.isosurface(volume, 0.90, yaw=math.pi / 2.0)

        print("[VLF] Inferring...")
        in_imgs = []
        for img in (img_pt, img_iso_1, img_iso_2, img_iso_6, img_iso_9):
            in_img = np.copy(img)
            in_img[:, :, 0:3] = in_img[:, :, 0:3] * 2 - 1
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)
            in_imgs.append(in_img)

        in_tensor = torch.cat(in_imgs, dim=0).unsqueeze(0)
        with torch.no_grad():
            out_tensor = model(in_tensor)

        out_img = out_tensor.squeeze(0).permute(1, 2, 0).numpy()

        print("[VLF] Saving...")
        for name, data in [
            ("img_pt.png", img_pt),
            ("img_iso_1.png", img_iso_1),
            ("img_iso_2.png", img_iso_2),
            ("img_iso_6.png", img_iso_6),
            ("img_iso_9.png", img_iso_9),
            ("img.png", out_img),
        ]:
            Image.fromarray((data * 255).astype(np.uint8)).save(
                config.working_dir / name
            )

        print("[VLF] Done.")


@cli.command()
def prepare(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option("--samples", "-s")] = 16,
    validation_samples: Annotated[int, typer.Option("--validation", "-l")] = 8,
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

            for j in range(samples):
                index = samples * i + j
                yaw = random.uniform(0, 2 * math.pi)
                print(
                    f"[VLF] Sample {index + 1}/{samples * len(assets)} (yaw={yaw:.2f})..."
                )
                create_sample(index, False, dataset_dir, volume, yaw, tracer)

            for j in range(validation_samples):
                index = validation_samples * i + j
                yaw = random.uniform(0, 2 * math.pi)
                print(
                    f"[VLF] Validation {index + 1}/{validation_samples * len(assets)} (yaw={yaw:.2f})..."
                )
                create_sample(index, True, dataset_dir, volume, yaw, tracer)

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

    model = PathTracerModel()
    if os.path.exists(config.working_dir / "model.pth"):
        model.load_state_dict(torch.load(config.working_dir / "model.pth"))

    model.to(device)

    loss_fn = nn.MSELoss()
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
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"[VLF] Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"LR: {scheduler._last_lr[0]:.8f}"
        )

        torch.save(model.state_dict(), config.working_dir / "model.pth")

    print("[VLF] Done.")


def main():
    cli()
