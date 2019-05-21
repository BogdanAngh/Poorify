import torch
import sys

sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from layers.recurrence import Recurrence

rec = Recurrence(105, 109)
inputs = torch.empty(1,1,105)
results = rec(inputs)

assert results.shape[0] == 1 and results.shape[1] == 109, 'Invalid shape!'

print('Test finished successfully!')