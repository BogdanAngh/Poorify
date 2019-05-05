import torch
import pandas as pd
from utils import find_emotion
from visualizer import plot_confusion_matrix

model = torch.load('my_model.pth')

data = pd.read_csv('test_mel_csv.csv')

cnt = 0
tot = 0

cm = torch.zeros(6, 6)

for e in data.itertuples():
    result = model.input_example(e.lyrics)
    if result == e.emotion:
        cnt += 1
    else:
        print(tot, ' ', result, ' vs ', e.emotion)
    tot += 1
    cm[find_emotion(e.emotion), find_emotion(result)] += 1

    label_names = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
    
print(cm)
plot_confusion_matrix(cm.numpy(), label_names)
print(cnt / tot)