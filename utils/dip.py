import torch
import torch.nn as nn
from itertools import pairwise


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, skip_channels, upsampling, last=False
    ):
        super().__init__()

        assert upsampling == "nearest" or upsampling == "bilinear"

        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels + skip_channels),
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2) if upsampling == "bilinear" else nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU() if not last else nn.Sigmoid(),
        )

    def forward(self, x, skip):
        if skip is not None:
            out = torch.cat((x, skip), dim=1)
        else:
            out = x

        return self.seq(out)


class Skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()

        self.no_out = out_channels == 0

        if not self.no_out:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel, padding=kernel // 2
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.no_out:
            return None

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DIP(nn.Module):
    def __init__(
        self,
        img_size,
        output_channels,
        z_channels,
        z_scale,
        z_noise,
        filters_down,
        filters_up,
        kernels_down,
        kernels_up,
        filters_skip,
        kernels_skip,
        upsampling
    ):
        super().__init__()
        self.size = img_size

        self.z_channels = z_channels

        self.output_channels = output_channels

        self.z_noise = z_noise

        last = [False] * len(filters_up)
        last[-1] = True

        self.shrink = nn.ModuleList(
            [
                Down(i, o, k)
                for (i, o), k in zip(
                    pairwise([self.z_channels] + filters_down), kernels_down
                )
            ]
        )

        self.grow = nn.ModuleList(
            [
                Up(i, o, k, s, last=l, upsampling=upsampling)
                for (i, o), k, l, s in zip(
                    pairwise(reversed([self.output_channels] + filters_up)),
                    reversed(kernels_up),
                    last,
                    reversed(filters_skip),
                )
            ]
        )
        self.skip = nn.ModuleList(
            [
                Skip(i, o, k)
                for (i, o, k) in zip(filters_down, filters_skip, kernels_skip)
            ]
        )

        z = torch.rand((1, self.z_channels, self.size, self.size)) * z_scale
        self.register_buffer("input", z)

    def forward(self):
        if self.z_noise != 0:
            noise = torch.randn( 1, self.z_channels, self.size, self.size, device=self.input.device) * self.z_noise
            out = self.input + noise
        else:
            out = self.input

        skip = []
        for l, sk in zip(self.shrink, self.skip):
            out = l(out)
            skip.append(sk(out))

        for l, sk in zip(self.grow, reversed(skip)):
            out = l(out, sk)

        return out

