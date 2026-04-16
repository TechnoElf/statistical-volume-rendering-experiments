import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
        for i in range(5):
            file_path = Path(self.directory) / (self.samples[idx] + f"_in{i}.png")
            in_img = Image.open(file_path).convert("RGBA")
            in_img = np.array(in_img)
            in_img = in_img.astype(np.float32) / 255.0
            in_img[:, :, 0:3] = (in_img[:, :, 0:3] * 2) - 1
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)
            in_imgs.append(in_img)

        in_imgs = torch.cat(in_imgs)
        return (in_imgs, out_img)
