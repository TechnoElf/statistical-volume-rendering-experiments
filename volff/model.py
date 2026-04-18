import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
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
        self.inc = DoubleConv(20, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)  # 512 x 512 x 64
        x2 = self.down1(x1)  # 256 x 256 x 128
        x3 = self.down2(x2)  # 128 x 128 x 256
        x4 = self.down3(x3)  # 64 x 64 x 512
        x5 = self.down4(x4)  # 32 x 32 x 1024
        x7 = self.up1(x5, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        out = self.outc(x10)
        tone_mapped_out = torch.sigmoid(out)
        mask = (x[:, 7, :, :] == 0).unsqueeze(1).expand_as(tone_mapped_out)
        masked_out = torch.where(
            mask,
            torch.tensor(
                0.0909, dtype=tone_mapped_out.dtype, device=tone_mapped_out.device
            ),
            tone_mapped_out,
        )
        return masked_out
