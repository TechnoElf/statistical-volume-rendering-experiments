# Reconstructed from: volff/cli/main.py
# Source: run/main.cpython-313.pyc

import math  # line 1
import os  # line 2
import random  # line 3
import subprocess  # line 4
from pathlib import Path  # line 5
from typing import Annotated  # line 6
from urllib.request import urlretrieve  # line 7

import numpy as np  # line 9
import torch  # line 10
import typer  # line 12
from PIL import Image  # line 13
from rich import print  # line 14
from torch import nn  # line 11
from torch.utils.data import DataLoader  # line 15

from volff.constants import asset_sources  # line 17
from volff.dataset import PathTracerDataset  # line 18
from volff.model import PathTracerModel  # line 19
from volff.trace import Tracer  # line 20
from volff.volume import load_vdb  # line 21

cli = typer.Typer()  # line 23


class Config:  # line 26
    working_dir: Path  # line 27

    def __init__(self, working_dir: Path):  # line 29
        self.working_dir = working_dir  # line 30


@cli.callback()  # line 33
def config(
    ctx: typer.Context,
    working_dir: Annotated[Path, typer.Option("--working-dir", "-w")] = Path("./run/"),
):
    ctx.obj = Config(working_dir=working_dir)  # line 38


@cli.command()  # line 41
def gather(ctx: typer.Context):
    config = ctx.obj  # line 43
    assets_dir = config.working_dir / "assets"  # line 44
    os.makedirs(assets_dir, exist_ok=True)  # line 45

    for name, url in asset_sources.items():  # line 47
        print(f"[VLF] Retrieving {name}...")  # line 48
        urlretrieve(url, assets_dir / name)  # line 49

    for name, url in asset_sources.items():  # line 51
        print(f"[VLF] Processing {name}...")  # line 52
        subprocess.run(["pvmdds", assets_dir / name])  # line 53
        subprocess.run(
            ["python", "scripts/pvm_to_vdb.py", assets_dir / name]
        )  # line 54

    print("[VLF] Done.")  # line 56


@cli.command()  # line 59
def trace(ctx: typer.Context):
    config = ctx.obj  # line 61
    assets_dir = config.working_dir / "assets"  # line 62
    os.makedirs(assets_dir, exist_ok=True)  # line 63

    with Tracer.create(1280, 720) as tracer:  # line 65
        print("[VLF] Loading volume...")  # line 66
        volume = load_vdb(assets_dir / "MRI-Head.vdb")  # line 67

        print("[VLF] Pathtracing...")  # line 69
        img = tracer.trace(volume, 256, yaw=math.pi / 2.0)  # line 70

        print("[VLF] Saving...")  # line 72
        img_uint8 = (img * 255).astype(np.uint8)  # line 73
        Image.fromarray(img_uint8).save(config.working_dir / "img_ref.png")  # line 74


@cli.command()  # line 77
def infer(ctx: typer.Context):
    config = ctx.obj  # line 79
    assets_dir = config.working_dir / "assets"  # line 80
    os.makedirs(assets_dir, exist_ok=True)  # line 81

    model = PathTracerModel()  # line 83
    model.load_state_dict(torch.load(config.working_dir / "model.pth"))  # line 84

    with Tracer.create(1280, 720) as tracer:  # line 86
        print("[VLF] Loading volume...")  # line 87
        volume = load_vdb(assets_dir / "MRI-Head.vdb")  # line 88

        print("[VLF] Pathtracing...")  # line 90
        img_pt = tracer.trace(volume, 1, yaw=math.pi / 2.0)  # line 91

        print("[VLF] Raycasting...")  # line 93
        img_iso_1 = tracer.isosurface(volume, 0.1, yaw=math.pi / 2.0)  # line 94
        img_iso_2 = tracer.isosurface(volume, 0.25, yaw=math.pi / 2.0)  # line 95
        img_iso_6 = tracer.isosurface(volume, 0.65, yaw=math.pi / 2.0)  # line 96
        img_iso_9 = tracer.isosurface(volume, 0.9, yaw=math.pi / 2.0)  # line 97

        print("[VLF] Inferring...")  # line 99
        in_imgs = []  # line 100
        for img in (img_pt, img_iso_1, img_iso_2, img_iso_6, img_iso_9):  # line 101
            in_img = np.copy(img)  # line 102
            in_img[:, :, 0:3] = in_img[:, :, 0:3] * 2 - 1  # line 103
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)  # line 104
            in_imgs.append(in_img)  # line 105

        in_tensor = torch.cat(in_imgs, dim=0).unsqueeze(0)  # line 107
        with torch.no_grad():  # line 108
            out_tensor = model(in_tensor)  # line 109

        out_img = out_tensor.squeeze(0).permute(1, 2, 0).numpy()  # line 111
        out_img = (out_img + 1) / 2  # line 112

        ref_img = tracer.trace(volume, 256, yaw=math.pi / 2.0)  # line 114

        print("[VLF] Saving...")  # line 116
        for name, data in [  # line 117
            ("img_pt.png", img_pt),
            ("img_inferred.png", out_img),
            ("img_ref.png", ref_img),
        ]:
            img_uint8 = (data[:, :, 0:3] * 255).astype(np.uint8)  # line 122
            Image.fromarray(img_uint8).save(config.working_dir / name)  # line 123

        print("[VLF] Done.")  # line 125


@cli.command()  # line 129
def prepare(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option("--samples", "-s")] = 16,
    validation_samples: Annotated[int, typer.Option("--validation", "-l")] = 8,
):
    config = ctx.obj  # line 135
    assets_dir = config.working_dir / "assets"  # line 136
    os.makedirs(assets_dir, exist_ok=True)  # line 137

    with Tracer.create(1280, 720) as tracer:  # line 139
        print("[VLF] Loading volume...")  # line 140
        volume = load_vdb(assets_dir / "MRI-Head.vdb")  # line 141

        print("[VLF] Preparing training data...")  # line 143
        train_dir = config.working_dir / "train"  # line 144
        os.makedirs(train_dir, exist_ok=True)  # line 145

        for i in range(samples):  # line 147
            yaw = random.uniform(0, 2 * math.pi)  # line 148
            print(f"[VLF] Sample {i + 1}/{samples} (yaw={yaw:.2f})...")  # line 149

            img_pt = tracer.trace(volume, 1, yaw=yaw)  # line 151
            img_ref = tracer.trace(volume, 256, yaw=yaw)  # line 152

            img_iso_1 = tracer.isosurface(volume, 0.1, yaw=yaw)  # line 154
            img_iso_2 = tracer.isosurface(volume, 0.25, yaw=yaw)  # line 155
            img_iso_6 = tracer.isosurface(volume, 0.65, yaw=yaw)  # line 156
            img_iso_9 = tracer.isosurface(volume, 0.9, yaw=yaw)  # line 157

            np.savez(  # line 159
                train_dir / f"sample_{i:04d}.npz",
                pt=img_pt,
                ref=img_ref,
                iso_1=img_iso_1,
                iso_2=img_iso_2,
                iso_6=img_iso_6,
                iso_9=img_iso_9,
            )

        print("[VLF] Preparing validation data...")  # line 169
        val_dir = config.working_dir / "val"  # line 170
        os.makedirs(val_dir, exist_ok=True)  # line 171

        for i in range(validation_samples):  # line 173
            yaw = random.uniform(0, 2 * math.pi)  # line 174
            print(
                f"[VLF] Validation {i + 1}/{validation_samples} (yaw={yaw:.2f})..."
            )  # line 175

            img_pt = tracer.trace(volume, 1, yaw=yaw)  # line 177
            img_ref = tracer.trace(volume, 256, yaw=yaw)  # line 178

            img_iso_1 = tracer.isosurface(volume, 0.1, yaw=yaw)  # line 179
            img_iso_2 = tracer.isosurface(volume, 0.25, yaw=yaw)  # line 180
            img_iso_6 = tracer.isosurface(volume, 0.65, yaw=yaw)  # line 181
            img_iso_9 = tracer.isosurface(volume, 0.9, yaw=yaw)  # line 182

            np.savez(  # line 184
                val_dir / f"sample_{i:04d}.npz",
                pt=img_pt,
                ref=img_ref,
                iso_1=img_iso_1,
                iso_2=img_iso_2,
                iso_6=img_iso_6,
                iso_9=img_iso_9,
            )

    print("[VLF] Done.")  # line 189


@cli.command()  # line 191
def train(
    ctx: typer.Context,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 8,
):
    config = ctx.obj  # line 196
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # line 197
    print(f"[VLF] Using device: {device}")  # line 198

    model = PathTracerModel().to(device)  # line 200
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # line 201
    loss_fn = nn.L1Loss()  # line 202

    train_dataset = PathTracerDataset(config.working_dir / "train")  # line 204
    val_dataset = PathTracerDataset(config.working_dir / "val")  # line 205
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # line 206
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # line 207

    for epoch in range(epochs):  # line 209
        model.train()  # line 210
        train_loss = 0.0  # line 211

        for batch in train_loader:  # line 213
            inputs = batch["input"].to(device)  # line 214
            targets = batch["target"].to(device)  # line 215

            optimizer.zero_grad()  # line 217
            outputs = model(inputs)  # line 218
            loss = loss_fn(outputs, targets)  # line 219
            loss.backward()  # line 220
            optimizer.step()  # line 221

            train_loss += loss.item()  # line 223

        train_loss /= len(train_loader)  # line 225

        model.eval()  # line 227
        val_loss = 0.0  # line 228
        with torch.no_grad():  # line 229
            for batch in val_loader:  # line 230
                inputs = batch["input"].to(device)  # line 231
                targets = batch["target"].to(device)  # line 232
                outputs = model(inputs)  # line 233
                loss = loss_fn(outputs, targets)  # line 234
                val_loss += loss.item()  # line 235

        val_loss /= len(val_loader)  # line 237

        print(  # line 239
            f"[VLF] Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f}"
        )

    # Save model  # line 245
    torch.save(model.state_dict(), config.working_dir / "model.pth")  # line 246
    print("[VLF] Model saved.")  # line 247

    # Save validation images  # line 249
    model.eval()  # line 250
    with torch.no_grad():  # line 251
        for i, batch in enumerate(val_loader):  # line 252
            inputs = batch["input"].to(device)  # line 253
            outputs = model(inputs)  # line 254
            out_img = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # line 255
            out_img = (out_img + 1) / 2  # line 256
            img_uint8 = (out_img * 255).astype(np.uint8)  # line 257
            Image.fromarray(img_uint8).save(
                config.working_dir / f"val_{i:04d}.png"
            )  # line 258

    print("[VLF] Done.")  # line 259


def main():  # line 261
    cli()
