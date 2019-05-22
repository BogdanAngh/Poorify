import torch
import torch.nn as nn

class PaddedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.padding = padding
        self.padder = nn.ConstantPad1d(self.padding, 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-3)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.padder(x)
        return self.activation(self.bn(self.conv(x)))