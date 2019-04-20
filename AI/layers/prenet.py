import torch
import torch.nn as nn

class Prenet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.in_size  = in_size
        #self.out_size = out_size
        #self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=0.5)
        self.linear  = nn.Linear(in_features = 256, out_features = 128)
        self.linear2 = nn.Linear(in_features = 128, out_features = 128)

        self.activation = nn.ReLU()

    def forward(self, x, is_training=True):
        
        x = self.activation(self.dropout(self.linear(x)))
        x = self.activation(self.dropout(self.linear2(x)))

        return x
