import numpy as np
import torch
from torch import nn


def conv(inputs, outputs, kernel=3):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, padding=kernel//2), nn.GroupNorm(1, outputs), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, channels=3, width=32):
        super().__init__()
        self.layers = nn.Sequential(conv(channels, width), conv(width, width), conv(width, width))

    def forward(self, x):
        return self.layers(x.float())


def warp(x, flow):
    """Bilinear border warp supporting the unrolled image-step derivatives."""
    n, channels, height, width = x.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=x.device, dtype=x.dtype),
                            torch.arange(width, device=x.device, dtype=x.dtype), indexing="ij")
    sx = (xx[None] + flow[:, 0]).clamp(0, width - 1)
    sy = (yy[None] + flow[:, 1]).clamp(0, height - 1)
    x0, y0 = sx.floor().long(), sy.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx, wy = sx - x0, sy - y0
    def sample(ix, iy):
        index = (iy * width + ix).flatten(1)[:, None].expand(-1, channels, -1)
        return x.flatten(2).gather(2, index).reshape(n, channels, height, width)
    top = (1 - wx[:, None]) * sample(x0, y0) + wx[:, None] * sample(x1, y0)
    bottom = (1 - wx[:, None]) * sample(x0, y1) + wx[:, None] * sample(x1, y1)
    return (1 - wy[:, None]) * top + wy[:, None] * bottom


def image_demo(seed, n, channels=3, mask=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, channels, 32, 32)).astype("float32")
    y = (x[:, :1] > .5).astype("float32") if mask else x.copy()
    return {"x": x, "y": y, "group": np.arange(n)}
