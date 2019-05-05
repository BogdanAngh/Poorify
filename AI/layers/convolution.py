import torch
import torch.nn as nn

#in-house imports
from layers.paddedconv import PaddedConvolution

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, max_len):
        super().__init__()

        self.conv = nn.ModuleList([
            PaddedConvolution(
                in_channels,
                out_channels,
                k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
            ) for k in range(2, 5)
        ])

        self.pooling = nn.Sequential(
            nn.ConstantPad1d([0, 1], 0),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):

        outs = []
        for c in self.conv:
            out = c(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)

        x = self.pooling(x)
        return x