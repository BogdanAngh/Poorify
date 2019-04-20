import torch
import torch.nn as nn

class Postnet(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2):
        super().__init__()

        self.in_size  = in_size
        self.out_size = out_size
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear  = nn.Linear(in_features = self.in_size, out_features = self.out_size)

        self.activation = nn.Tanh()

    # multiply by 3 to get results in (-3, 3) range
    def forward(self, x, is_training=True):
        
        if is_training == False:
            self.dropout.p = 0.0

        return (self.activation(self.dropout(self.linear(x))) * 3)