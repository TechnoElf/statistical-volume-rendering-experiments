import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PathTracerDataset(Dataset):
    def __init__(self, train=False, directory="dataset"):
        self.directory = directory
        files = sorted(os.listdir(directory))
        ids = [file.split(".")[0] for file in files if file.endswith(".png")]
        ids = [id.split("_")[1] for id in ids if id.startswith("pt_")]

        if train:
            ids = ids[len(ids) // 4 :]
        else:
            ids = ids[: len(ids) // 4]

        ids = np.random.permutation(ids)

        self.output = [f"pt128s_{id}.png" for id in ids]
        self.inputs = [
            [f"pt1s_{id}.png"] + [f"rc{i:02d}_{id}.png" for i in [1, 2, 6, 9]]
            for id in ids
        ]

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        file_path = Path(self.directory) / self.output[idx]
        out_img = Image.open(file_path).convert("RGB")
        out_img = np.array(out_img)
        out_img = out_img.astype(np.float32) / 255.0
        out_img = torch.from_numpy(out_img).permute(2, 0, 1)

        in_imgs = []
        for input_file in self.inputs[idx]:
            file_path = Path(self.directory) / input_file
            in_img = Image.open(file_path).convert("RGBA")
            in_img = np.array(in_img)
            in_img = in_img.astype(np.float32) / 255.0
            in_img[:, :, 0:3] = (in_img[:, :, 0:3] * 2) - 1
            in_img = torch.from_numpy(in_img).permute(2, 0, 1)
            in_imgs.append(in_img)

        # x_pos_img = torch.from_numpy(
        #     np.tile(
        #         np.linspace(0, 1, in_imgs[0].shape[2]).astype(np.float32)
        #         / in_imgs[0].shape[2],
        #         [in_imgs[0].shape[1], 1],
        #     )
        # )
        x_pos_img = torch.from_numpy(
            np.ones((in_imgs[0].shape[1], in_imgs[0].shape[2]), dtype=np.float32)
        )

        # y_pos_img = torch.from_numpy(
        #     np.tile(
        #         np.linspace(0, 1, in_imgs[0].shape[1]).astype(np.float32)
        #         / in_imgs[0].shape[1],
        #         [in_imgs[0].shape[2], 1],
        #     )
        # ).T
        y_pos_img = torch.from_numpy(
            np.ones((in_imgs[0].shape[1], in_imgs[0].shape[2]), dtype=np.float32)
        )

        in_imgs.append(torch.stack((x_pos_img, y_pos_img)))

        in_imgs = torch.cat(in_imgs)
        return (in_imgs, out_img)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PathTracerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(22, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        # self.bottleneck = nn.Sequential(
        #    nn.Flatten(),
        #    nn.Linear(64 * 80 * 45, 256),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(256, 64 * 80 * 45),
        #    nn.ReLU(inplace=True),
        #    nn.Unflatten(1, (64, 80, 45)),
        # )
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.bottleneck(x5)
        x7 = self.up1(x5, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        return torch.sigmoid(self.outc(x10))
