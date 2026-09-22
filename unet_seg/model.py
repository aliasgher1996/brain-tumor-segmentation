"""U-Net (Ronneberger et al., 2015) with padded convolutions and batch normalisation,
exactly as defined in the original notebook."""
import torch
from torch import nn


class DoubleConvBlock(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) x 2, padding 1 so spatial size is preserved."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder widths ``block_sizes`` (default 64-128-256-512), bottleneck 2x the last width,
    transposed-conv upsampling with skip connections, 1x1 output conv producing logits."""

    def __init__(self, in_channels=3, out_channels=1, block_sizes=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = nn.ModuleList()
        ch = in_channels
        for b in block_sizes:
            self.encoder.append(DoubleConvBlock(ch, b))
            ch = b
        self.bottleneck = DoubleConvBlock(block_sizes[-1], 2 * block_sizes[-1])
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for b in reversed(block_sizes):
            self.up.append(nn.ConvTranspose2d(2 * b, b, kernel_size=2, stride=2))
            self.decoder.append(DoubleConvBlock(2 * b, b))
        self.head = nn.Conv2d(block_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            x = dec(torch.cat((skip, x), dim=1))
        return self.head(x)


def build_model(cfg):
    return UNet(cfg.data.in_channels, cfg.model.out_channels, tuple(cfg.model.block_sizes))
