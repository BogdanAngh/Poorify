import torch
import torch.nn as nn

#in-house imports
from layers.paddedconv import PaddedConvolution

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, max_len):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len = max_len

        self.conv = nn.ModuleList([
            PaddedConvolution(
                self.in_channels,
                self.out_channels,
                k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
            ) for k in range(2, 5)
        ])

        self.pooling = nn.Sequential(
            nn.ConstantPad1d([0, 1], 0),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        #self.conv = nn.Conv1d(in_channels = self.in_channels, out_channels = self.out_channels,
        #                      kernel_size = self.kernel_size)
        #self.activation = nn.ReLU()
        #self.pooling = nn.MaxPool1d(self.max_len - self.kernel_size + 1)
        #self.pooling = nn.MaxPool1d(2, stride=1)

    def forward(self, x):

        outs = []
        for c in self.conv:
            out = c(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)

        x = self.pooling(x)

        # in  = batch_size x in_channels x embedding_size
        # out = batch_size x out_channels x embedding_size - kernel_size + 1
        #x = self.conv(x)

        # applies ReLU over the convolution
        #x = self.activation(x)
        
        # in = batch_size x out_channels x embedding_size - kernel_size
        # out = batch_size x out_channels x 1
        #x = self.pooling(x)
        
        #returns batch_size x out_channels (one feature for each map)
        return x