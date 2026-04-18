import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TILE_SIZE = 512
HALF_TILE_SIZE = int(TILE_SIZE / 2)
QUARTER_TILE_SIZE = int(HALF_TILE_SIZE / 2)


def random_sample(index, validation, dir, volume, tracer):
    pitch = random.uniform(-math.pi / 2, math.pi / 2)
    yaw = random.uniform(0, 2 * math.pi)
    roll = random.uniform(0, 2 * math.pi)
    spp = random.randrange(1, 5)
    tile_start_x = random.randrange(1280 - 512)
    tile_start_y = random.randrange(720 - 512)

    create_sample(
        index,
        validation,
        dir,
        volume,
        tracer,
        pitch,
        yaw,
        roll,
        spp,
        tile_start_x,
        tile_start_y,
    )


def create_sample(
    index,
    validation,
    dir,
    volume,
    tracer,
    pitch,
    yaw,
    roll,
    spp,
    tile_start_x,
    tile_start_y,
):
    name = f"v{index:07d}" if validation else f"{index:08d}"

    img_ref = tracer.trace(volume, 256, pitch=pitch, yaw=yaw, roll=roll)
    img_pt = tracer.trace(volume, spp, pitch=pitch, yaw=yaw, roll=roll)

    img_iso_1 = tracer.isosurface(volume, 0.10, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_2 = tracer.isosurface(volume, 0.25, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_6 = tracer.isosurface(volume, 0.65, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_9 = tracer.isosurface(volume, 0.90, pitch=pitch, yaw=yaw, roll=roll)

    img_ref, img_pt, img_iso_1, img_iso_2, img_iso_6, img_iso_9 = (
        img[
            tile_start_y : tile_start_y + TILE_SIZE,
            tile_start_x : tile_start_x + TILE_SIZE,
        ]
        for img in (img_ref, img_pt, img_iso_1, img_iso_2, img_iso_6, img_iso_9)
    )

    for path, data in [
        (f"{name}_in0.png", img_pt),
        (f"{name}_in1.png", img_iso_1),
        (f"{name}_in2.png", img_iso_2),
        (f"{name}_in3.png", img_iso_6),
        (f"{name}_in4.png", img_iso_9),
        (f"{name}_out.png", img_ref),
    ]:
        Image.fromarray((data * 255).astype(np.uint8)).save(dir / path)


def tile_image(img):
    height = img.shape[0]
    width = img.shape[1]

    img = np.pad(
        img,
        ((QUARTER_TILE_SIZE, TILE_SIZE), (QUARTER_TILE_SIZE, TILE_SIZE), (0, 0)),
        mode="reflect",
    )

    tiles = []
    for y in range(0, height, HALF_TILE_SIZE):
        for x in range(0, width, HALF_TILE_SIZE):
            tile = img[y : y + TILE_SIZE, x : x + TILE_SIZE, :]
            tiles.append(tile)

    return tiles


def untile_image(tiles, width, height):
    n_y_tiles = int(math.ceil(height / HALF_TILE_SIZE))
    n_x_tiles = int(math.ceil(width / HALF_TILE_SIZE))

    img = np.zeros(
        (
            height + TILE_SIZE + QUARTER_TILE_SIZE,
            width + TILE_SIZE + QUARTER_TILE_SIZE,
            tiles[0].shape[2],
        ),
        dtype=tiles[0].dtype,
    )

    counts = np.zeros(
        (
            height + TILE_SIZE + QUARTER_TILE_SIZE,
            width + TILE_SIZE + QUARTER_TILE_SIZE,
            tiles[0].shape[2],
        )
    )

    filter = np.repeat(
        np.pad(
            np.ones((HALF_TILE_SIZE, HALF_TILE_SIZE)),
            QUARTER_TILE_SIZE,
            mode="linear_ramp",
        )[:, :, None],
        3,
        axis=2,
    )

    for y in range(n_y_tiles):
        for x in range(n_x_tiles):
            img[
                y * HALF_TILE_SIZE : (y + 2) * HALF_TILE_SIZE,
                x * HALF_TILE_SIZE : (x + 2) * HALF_TILE_SIZE,
                :,
            ] += tiles[y * n_x_tiles + x] * filter

            counts[
                y * HALF_TILE_SIZE : (y + 2) * HALF_TILE_SIZE,
                x * HALF_TILE_SIZE : (x + 2) * HALF_TILE_SIZE,
                :,
            ] += filter

    img = img[
        QUARTER_TILE_SIZE : QUARTER_TILE_SIZE + height,
        QUARTER_TILE_SIZE : QUARTER_TILE_SIZE + width,
    ]
    counts = counts[
        QUARTER_TILE_SIZE : QUARTER_TILE_SIZE + height,
        QUARTER_TILE_SIZE : QUARTER_TILE_SIZE + width,
    ]

    return np.divide(img, counts)


class PathTracerDataset(Dataset):
    def __init__(self, directory, train=False, random=True):
        self.directory = directory
        files = sorted(os.listdir(directory))
        samples = [file[:8] for file in files if file.endswith("out.png")]

        validation_samples = [sample for sample in samples if sample.startswith("v")]
        samples = [sample for sample in samples if sample.startswith("0")]
        if not train:
            samples = validation_samples

        if random:
            samples = np.random.permutation(samples)

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = Path(self.directory) / (self.samples[idx] + "_out.png")
        out_img = Image.open(file_path).convert("RGB")
        out_img = np.array(out_img)
        out_img = out_img.astype(np.float32) / 255.0
        out_img = torch.from_numpy(out_img).permute(2, 0, 1)

        in_imgs = []

        file_path = Path(self.directory) / (self.samples[idx] + f"_in0.png")
        in_img = Image.open(file_path).convert("RGBA")
        in_img = np.array(in_img)
        in_img = in_img.astype(np.float32) / 255.0
        in_img = torch.from_numpy(in_img).permute(2, 0, 1)
        in_imgs.append(in_img)

        for i in range(1, 5):
            file_path = Path(self.directory) / (self.samples[idx] + f"_in{i}.png")
            in_img = Image.open(file_path).convert("RGBA")
            in_img = np.array(in_img)
            in_img = in_img.astype(np.float32) / 255.0
            in_img[:, :, 0:3] = (in_img[:, :, 0:3] * 2) - 1
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)
            in_imgs.append(in_img)

        in_imgs = torch.cat(in_imgs)

        assert in_imgs.shape == torch.Size([20, 512, 512]), (
            f"data has wrong shape: {in_imgs.shape}"
        )
        assert out_img.shape == torch.Size([3, 512, 512]), (
            f"data has wrong shape: {out_img.shape}"
        )

        return (in_imgs, out_img)
