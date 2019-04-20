import torch
import torch.nn as nn

class PaddedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.padding = padding
        self.padder = nn.ConstantPad1d(self.padding, 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.padder(x)
        return self.activation(self.conv(x))
