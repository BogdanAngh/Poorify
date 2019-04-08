import lyricwikia
import re
import os
import errno
import json
import datetime
import pandas as pd
import numpy as np
import logging
import torch

from shutil import copyfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from attrdict import AttrDict

#in-house libraries
from dataset import MyDataset
from vocabulary import Vocabulary
import constants

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

    logging.info('Loaded {} samples'.format(data.shape[0]))

    #transform the dataframe into (sample, label) pairs
    #for the moment samples is a numpy array
    samples = data['lyrics'].values
    valence = torch.tensor(data['valence'].values).to(constants.DEVICE)
    arousal = torch.tensor(data['arousal'].values).to(constants.DEVICE)

    #build the labels tensor : (valence, arousal) x input_size
    labels = torch.t(torch.stack((valence, arousal), 0)).view(-1, 2).to(constants.DEVICE)

    #build the vocabulary
    vocab = Vocabulary(''.join(str(e) for e in samples))
    logging.info('Vocab size : {}'.format(vocab.size()))

    #transform the samples into tensors and pad them to the maximum length
    samples = pad_sequence([text_to_tensor(e, vocab) for e in samples], batch_first=True).to(constants.DEVICE)

    train_size = int(samples.shape[0] * CONFIG['train_val_cutoff'])
    train_samples, validation_samples = samples[:train_size], samples[train_size:]
    train_labels, validation_labels = labels[:train_size], labels[train_size:]

    #build the training data loader
    train_dataset = MyDataset(train_samples, train_labels)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=CONFIG['batch_size'], 
                                              shuffle=CONFIG['shuffle'], 
                                              num_workers=CONFIG['num_workers'])

    logging.info('Created training data loader having {} samples and {} batches of size {}'.format(
        train_size, len(train_data_loader), CONFIG['batch_size']))

    #build the validation data loader
    validation_dataset = MyDataset(train_samples, train_labels)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, 
                                              batch_size=CONFIG['batch_size'], 
                                              shuffle=CONFIG['shuffle'], 
                                              num_workers=CONFIG['num_workers'])

    logging.info('Created validation data loader having {} samples and {} batches of size {}'.format(
        samples.shape[0] - train_size, len(validation_data_loader), CONFIG['batch_size']))

    return vocab, train_data_loader, validation_data_loader

def load_config(config_path):

    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()

    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)

    logging.info('Config loaded!')

    return config

def load_logging():

    #change the logging format
    logger = logging.getLogger()
    logging.basicConfig(format="[%(asctime)s | %(levelname)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s", 
                        filename='app.log', filemode='a', datefmt='%d-%b-%y %H:%M:%S')

    #log any kind of information
    logger.setLevel(logging.DEBUG)

    logging.info('Logger loaded')

def save_model(model):

    log_path = 'log/'

    try:
        logging.info('Creating log directory {}'.format(log_path))
        #create the log directory
        os.mkdir(log_path)
    except FileExistsError:
        logging.warning('Log directory already exists!')
        pass

    try:
        now = datetime.datetime.now()
        #create an unique directory name
        log_path += str(now)

        #create the directory which holds the model
        logging.info('Creating model s directory {}'.format(log_path))
        os.mkdir(log_path)  
    except FileExistsError:
        logging.warning('Model s directory already exists!')
        pass

    #save the model in the created directory
    logging.info('Saving the model at {}'.format(log_path + '/my_model.pth'))
    torch.save(model.state_dict(), log_path + '/my_model.pth')

    #copy the used config
    logging.info('Saving the model s config at {}'.format(log_path + '/config.json'))
    copyfile('config.json', log_path + '/config.json')