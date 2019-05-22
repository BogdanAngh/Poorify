import requests
import pandas as pd
import random

import sys
sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from models.model import MyModel
from utils import preprocess, text_to_tensor
from emotionClasification import emotion_map

url = 'http://localhost:5002/api'

data = pd.read_csv('test_dataset.csv')
rnd = random.randint(0, 3112)
data = data.iloc[rnd]
print(data['artist_name'], ' ', data['track_name'])
lyrics = data['lyrics']

#preprocess
lyrics = preprocess(lyrics) 
#transform into tensor of indexes

r = requests.post(url,json={'input':lyrics})
print(r.json())