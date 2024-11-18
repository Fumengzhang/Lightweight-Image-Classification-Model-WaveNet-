import torch
from torch import nn
import torch.nn.functional as F


class BSConv(nn.Module):
    """Blueprint Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BSConv, self).__init__()
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.depthwise_conv = nn.Conv2d(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=out_channels)

    def forward(self, x):
        return self.depthwise_conv(self.pointwise_conv(x))


class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale=2, dropout=0.1):
        super().__init__()
        mid = int(out_channels * scale)
        self.same = (in_channels == out_channels) and (stride == 1)
        # self.upconv = BSConv(in_channels, int(in_channels * scale), kernel_size, stride, padding)
        self.blocks = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            BSConv(in_channels, mid, kernel_size, stride, padding),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.BatchNorm2d(mid),
            BSConv(mid, out_channels, 3, 1, 1),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # if self.same:
        #     x += self.blocks(x)
        # else:
        #     x = self.blocks(x)
        if self.same:
            # x += self.blocks(x)
            y = self.blocks(x)
            y = y * F.sigmoid(y) + x
        else:
            y = self.blocks(x)
        return y


class WaveStage(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[2, 3, 2, 2], dropout=0.1, reps=3):
        """reps denotes the number of repetitions of the 'same' UpDownBlock keeping the spatial dimension same,
        not including the neck one, which down-samples the feature maps and usually increases the output channels."""
        super().__init__()
        assert len(scales) == reps + 1, 'len(scale) != (reps + 1)'
        self.blocks = nn.Sequential(
            *[UpDownBlock(in_channels, in_channels, scale=scale, dropout=dropout) for scale in scales[:-1]]
        )
        self.neck = UpDownBlock(in_channels, out_channels, 3, 2, 1, scale=scales[-1])

    def forward(self, x):
        return self.neck(self.blocks(x) + x)


class WaveNet10(nn.Module):
    """resp=[6, 5, 4], dropout=0.1, scale=2"""
    def __init__(self, scales, reps=[7, 5, 3], dim=16, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            BSConv(3, dim, 3, 1, 1),  # dim x 32 x 32
            WaveStage(dim, 2 * dim, scales[0], dropout, reps[0]),
            WaveStage(2 * dim, 4 * dim, scales[1], dropout, reps[1]),
            WaveStage(4 * dim, 8 * dim, scales[2], dropout, reps[2]),
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(8 * dim, 10)
        )

    def forward(self, x):
        return self.blocks(x)

