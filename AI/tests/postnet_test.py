import torch
import sys

sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from layers.postnet import Postnet

postnet = Postnet(100, 50)
inputs = torch.empty(100)
results = postnet(inputs)

assert results.shape[0] == 50, 'Invalid shape!'

print('Test finished successfully!')