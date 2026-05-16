import torch
import torch.nn as nn

from volff.models.common import DoubleConv, Down, OutConv, Up


class SimplePathTracerDenoiseModel(nn.Module):
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
        self.gamma = nn.Parameter(torch.tensor(2.2))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
        tone_mapped_out = torch.sigmoid(out).pow(self.gamma)
        mask = (x[:, 7, :, :] == 0).unsqueeze(1).expand_as(tone_mapped_out)
        masked_out = torch.where(
            mask,
            torch.tensor(
                0.0909, dtype=tone_mapped_out.dtype, device=tone_mapped_out.device
            ),
            tone_mapped_out,
        )
        return masked_out
