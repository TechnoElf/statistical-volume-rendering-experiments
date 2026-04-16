import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def random_sample(index, validation, dir, volume, tracer):
    pitch = random.uniform(-math.pi / 2, math.pi / 2)
    yaw = random.uniform(0, 2 * math.pi)
    roll = random.uniform(0, 2 * math.pi)
    spp = random.randrange(1, 5)

    create_sample(index, validation, dir, volume, tracer, pitch, yaw, roll, spp)


def create_sample(index, validation, dir, volume, tracer, pitch, yaw, roll, spp):
    name = f"v{index:07d}" if validation else f"{index:08d}"

    img_ref = tracer.trace(volume, 256, pitch=pitch, yaw=yaw, roll=roll)
    img_pt = tracer.trace(volume, spp, pitch=pitch, yaw=yaw, roll=roll)

    img_iso_1 = tracer.isosurface(volume, 0.10, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_2 = tracer.isosurface(volume, 0.25, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_6 = tracer.isosurface(volume, 0.65, pitch=pitch, yaw=yaw, roll=roll)
    img_iso_9 = tracer.isosurface(volume, 0.90, pitch=pitch, yaw=yaw, roll=roll)

    for path, data in [
        (f"{name}_in0.png", img_pt),
        (f"{name}_in1.png", img_iso_1),
        (f"{name}_in2.png", img_iso_2),
        (f"{name}_in3.png", img_iso_6),
        (f"{name}_in4.png", img_iso_9),
        (f"{name}_out.png", img_ref),
    ]:
        Image.fromarray((data * 255).astype(np.uint8)).save(dir / path)


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
        return (in_imgs, out_img)
