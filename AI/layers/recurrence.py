import torch
import torch.nn as nn

class Recurrence(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size,
                          num_layers = 4, batch_first = True, bidirectional = False)

    def forward(self, x):

        result, _ = self.gru(x)
        #return result.contiguous().view(result.shape[0], 1, -1).squeeze(1)
        return result[:, result.shape[1] - 1, :]