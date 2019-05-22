import lyricwikia
import re
import os
import errno
import json
import string
import datetime
import pandas as pd
import numpy as np
import logging
import torch
from langdetect import detect

from shutil import copyfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from attrdict import AttrDict

#in-house libraries
from dataset import MyDataset
from vocabulary import Vocabulary
from emotionClasification import findQuadrant
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

    dataset.to_csv(result_path, index=False)

    print('{} samples failed to import'.format(fails))

def text_to_tensor(text, vocab):
    """
        Gets a text and converts it to a tensor of indexes found in the vocabulary
        'This is an example' => [10, 15, 20, 2]
    """ 
    return torch.tensor([vocab.word_to_idx.get(e, vocab.size() - 1) for e in text])

def tensor_to_text(x, vocab):
    """
        Gets a tensor of indexes and returns the corresponding text using the map from the given vocabulary
        [10, 15, 20, 2] => 'This is an example'
    """
    return ''.join(vocab.idx_to_word[e.item()] for e in x)

def preprocess(word):
    
    #use only lower cases
    word = word.lower()

    #remove numbers
    word = re.sub(r'\d+', '', word)

    #remove endlines
    word = re.sub('\n', ' ', word)
    
    #remove punctuation
    word = word.translate(str.maketrans('', '', string.punctuation))

    #remove unwanted white spaces
    word = word.strip()

    return word

def build_loader(sample, label, name, CONFIG):

    dataset = MyDataset(sample, label)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=CONFIG['batch_size'], 
                                              shuffle=CONFIG['shuffle'], 
                                              num_workers=CONFIG['num_workers'])

    logging.info('Created {} loader having {} samples and {} batches of size {}'.format(
        name, sample.shape[0], len(data_loader), CONFIG['batch_size']))

    return data_loader

def load_train_val(train_path, val_path, CONFIG):

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    #delete useless columns
    train_data.drop('artist_name', 1)
    train_data.drop('track_name', 1)
    val_data.drop('artist_name', 1)
    val_data.drop('track_name', 1)

    #TRAIN_DATA : delete samples without lyrics
    train_data = train_data[train_data['lyrics'] != '-']
    logging.info('Loaded {} samples from {}'.format(train_data.shape[0], train_path))

    #VALIDATION_DATA : delete samples without lyrics
    val_data = val_data[val_data['lyrics'] != '-']
    logging.info('Loaded {} samples from {}'.format(val_data.shape[0], val_path))

    #TRAIN_DATA : build the samples & labels array
    train_samples = [preprocess(e) for e in train_data['lyrics'].values]
    valence = torch.tensor(train_data['valence'].values, dtype=torch.float32)
    arousal = torch.tensor(train_data['arousal'].values, dtype=torch.float32)

    #TRAIN_DATA : build the labels tensor
    train_labels = torch.tensor([findQuadrant(l[0], l[1]) for l in zip(valence, arousal)])

    print(torch.sum(train_labels == 0))
    print(torch.sum(train_labels == 1))
    print(torch.sum(train_labels == 2))
    print(torch.sum(train_labels == 3))

    #VALIDATION_DATA : build the samples & labels array
    val_samples = [preprocess(e) for e in val_data['lyrics'].values]
    valence = torch.tensor(val_data['valence'].values, dtype=torch.float32)
    arousal = torch.tensor(val_data['arousal'].values, dtype=torch.float32)

    #VALIDATION_DATA : build the labels tensor
    val_labels = torch.tensor([findQuadrant(l[0], l[1]) for l in zip(valence, arousal)])

    #build the vocabulary
    samples = train_samples + val_samples
    vocab = Vocabulary(' '.join(str(e) for e in samples))
    vocab.add_unkw()

    #TRAIN_DATA : transform the samples into tensors and pad them to the maximum length
    train_samples = pad_sequence([text_to_tensor(e.split(), vocab) for e in train_samples], batch_first=True)
    #TRAIN_DATA : set the maximum length of an input to max_len
    train_samples = train_samples[:, :CONFIG['max_len']]

    #TRAIN_DATA : build the training data loader
    train_data_loader = build_loader(train_samples, train_labels, 'train data', CONFIG)

    #VALIDATION_DATA : transform the samples into tensors and pad them to the maximum length
    val_samples = pad_sequence([text_to_tensor(e.split(), vocab) for e in val_samples], batch_first=True)
    #VALIDATION_DATA : set the maximum length of an input to max_len
    val_samples = val_samples[:, :CONFIG['max_len']]

    #VALIDATION_DATA : build the training data loader
    val_data_loader = build_loader(val_samples, val_labels, 'validation data', CONFIG)

    return vocab, train_data_loader, val_data_loader

def load_test(path, vocab, CONFIG):

    data = pd.read_csv(path)

    #delete useless columns
    data.drop('artist_name', 1)
    data.drop('track_name', 1)

    #delete samples without lyrics
    data = data[data['lyrics'] != '-']
    logging.info('Loaded {} samples from {}'.format(data.shape[0], path))

    #transform the dataframe into (sample, label) pairs
    #for the moment samples is a numpy array
    samples = [preprocess(e) for e in data['lyrics'].values]
    valence = torch.tensor(data['valence'].values, dtype=torch.float32)
    arousal = torch.tensor(data['arousal'].values, dtype=torch.float32)

    #build the labels tensor
    labels = torch.tensor([findQuadrant(l[0], l[1]) for l in zip(valence, arousal)])

    #transform the samples into tensors and pad them to the maximum length
    samples = pad_sequence([text_to_tensor(e.split(), vocab) for e in samples], batch_first=True)
    #set the maximum length of an input to max_len
    samples = samples[:, :CONFIG['max_len']]

    test_data_loader = build_loader(samples, labels, 'test data', CONFIG)
    return test_data_loader

def load_data(CONFIG=None):

    #build the data loaders
    #build train&validation in the same function to append the inputs and create the vocabulary
    vocab, train_data_loader, validation_data_loader = load_train_val(constants.TRAIN_PATH, constants.VALIDATION_PATH, CONFIG)
    #won't add the test words in vocabulary
    #for unknown words use special word UNKW
    test_data_loader = load_test(constants.TEST_PATH, vocab, CONFIG)
    logging.info('Vocab size : {}'.format(vocab.size()))

    return vocab, train_data_loader, validation_data_loader, test_data_loader

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

def save_model(model, conf_matrix):

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
    torch.save(model, log_path + '/my_model.pth')

    #save the confusion matrix
    logging.info('Saving the confusion matrix at {}'.format(log_path + '/cm.png'))
    conf_matrix.savefig(log_path + '/cm.png')

    #copy the used config
    logging.info('Saving the model s config at {}'.format(log_path + '/config.json'))
    copyfile('config.json', log_path + '/config.json')

def confusion_matrix(predicted, label, no_of_classes):

    conf_matrix = torch.zeros(no_of_classes, no_of_classes)
    for p, l in zip(predicted, label):
        conf_matrix[l, p] += 1

    return conf_matrix