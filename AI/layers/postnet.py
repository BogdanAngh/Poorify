import torch
import torch.nn as nn

class Postnet(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear  = nn.Linear(in_features = in_size, out_features = out_size)

        #CrossEntropyLoss takes the linear output
        self.activation = None

    def forward(self, x, is_training=True):
        
        #don t use dropout in validation phase
        if is_training == False:
            self.dropout.p = 0.0
        else:
            self.dropout.p = self.dropout_rate

        if self.activation is not None:
            x = self.activation(self.dropout(self.linear(x)))
        else:
            x = self.dropout(self.linear(x))

        return x