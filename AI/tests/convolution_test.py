import torch
import sys

sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from layers.convolution import Convolution

# TEST 1
conv = Convolution(1, 1, 30)
inputs = torch.empty(1,1,30)
result = conv(inputs)

assert result.shape[0] == 1 and result.shape[1] == 3 and result.shape[2] == 15, 'Invalid shape in TEST 1!'

# TEST 2
conv = Convolution(2, 1, 30)
inputs = torch.empty(1,2,30)
result = conv(inputs)

assert result.shape[0] == 1 and result.shape[1] == 3 and result.shape[2] == 15, 'Invalid shape in TEST 2!'

# TEST 3
conv = Convolution(1, 2, 30)
inputs = torch.empty(1,1,30)
result = conv(inputs)

assert result.shape[0] == 1 and result.shape[1] == 6 and result.shape[2] == 15, 'Invalid shape in TEST 3!'

print('Test finished successfully!')

