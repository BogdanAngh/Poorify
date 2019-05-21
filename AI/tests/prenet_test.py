import torch
import sys

sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from layers.prenet import Prenet

prenet = Prenet()
inputs = torch.empty(256)
results = prenet(inputs)

assert results.shape[0] == 128, 'Invalid shape!'

print('Test finished successfully!')