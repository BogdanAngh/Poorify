import lyricwikia
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from attrdict import AttrDict

#in-house libraries
from dataset import MyDataset
from vocabulary import Vocabulary

def api_result(status):
    if status == True:
        return 'Succes'

    return 'Fail'

def get_lyrics(initial_path, result_path):

    dataset = pd.read_csv(initial_path)
    dataset['lyrics'] = '-'
    lyrics_threeshold = 4

    print('Starting the lyrics reader...')
    fails = 0
    for idx, data in dataset.iterrows():
        artist = data['artist_name']
        track = data['track_name']
        has_failed = True
        try:
            lyrics = lyricwikia.get_lyrics(data['artist_name'], data['track_name']).split('\n')
            lyrics.remove('')

            #get each line with its frequency
            lyrics, counts = np.unique(lyrics, return_counts=True)

            #sort the lyrics by their frequency
            count_sort_ind = np.argsort(-counts)

            #set the lyrics in the dataframe
            dataset.loc[idx, 'lyrics'] = '\n'.join(lyrics[count_sort_ind[1:1 + lyrics_threeshold]])
        except Exception as e:
            print('Couldn t get {} - {} at index {} with error {}'.format(artist, track, idx, e))
            fails = fails + 1
            has_failed = False

        print('{}. {} - {} : {}'.format(idx, artist, track, api_result(has_failed)))
        
        if idx == 10:
            break

    dataset.to_csv(result_path, index=False)

    print('{} samples failed to import'.format(fails))

def text_to_tensor(text, vocab):
    """
        Gets a text and converts it to a tensor of indexes found in the vocabulary
        'Example' => [10, 15, 20, 2, 50, 45, 44]
    """ 
    return torch.tensor([vocab.char_to_idx[e] for e in text])

def tensor_to_text(x, vocab):
    """
        Gets a tensor of indexes and returns the corresponding text using the map from the given vocabulary
        [10, 15, 20, 2, 50, 45, 44] => 'Example'
    """
    return ''.join(vocab.idx_to_chr[e.item()] for e in x)

def load_data(path, CONFIG=None):

    data = pd.read_csv(path)

    #delete useless columns
    data.drop('artist_name', 1)
    data.drop('track_name', 1)

    #delete samples without lyrics
    data = data[data['lyrics'] != '-']

    #transform the dataframe into (samples, labels) pair
    #for the moment the samples is a numpy array
    samples = data['lyrics'].values
    valence = torch.tensor(data['valence'].values)
    arousal = torch.tensor(data['arousal'].values)

    #build the labels tensor : (valence, arousal) x input_size
    labels = torch.t(torch.stack((valence, arousal), 0)).view(-1, 2)

    #build the vocabulary
    vocab = Vocabulary(''.join(str(e) for e in samples))
    print('Vocab size : ', vocab.size())

    #transform the samples into tensors and pad them to the maximum length
    samples = pad_sequence([text_to_tensor(e, vocab) for e in samples], batch_first=True)

    my_dataset = MyDataset(samples, labels)
    #TODO : use CONFIG data
    data_loader = torch.utils.data.DataLoader(my_dataset, 
                                              batch_size=CONFIG['batch_size'], 
                                              shuffle=CONFIG['shuffle'], 
                                              num_workers=CONFIG['num_workers'])

    return data_loader

def load_config(config_path):

    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()

    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)

    return config