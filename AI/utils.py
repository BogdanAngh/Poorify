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

idx_to_emotion = dict({0: 'joy', 1 : 'sadness', 2 : 'anger', 3 : 'fear', 4 : 'love', 5 : 'surprise'})

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

def find_emotion(e):
    if e == 'joy':
        return 0
    elif e == 'sadness':
        return 1
    elif e == 'anger':
        return 2
    elif e == 'fear':
        return 3
    elif e == 'love':
        return 4
    elif e == 'surprise':
        return 5

    return -1

def load_train_val(train_samples, train_labels, val_samples, val_labels, CONFIG):

    #TRAIN_DATA : build the samples & labels array
    train_samples = [preprocess(e) for e in train_samples.values]
    train_labels = torch.tensor([find_emotion(e) for e in train_labels])
    print('Train samples preprocessed!')

    #VALIDATION_DATA : build the samples & labels array
    val_samples = [preprocess(e) for e in val_samples.values]
    val_labels = torch.tensor([find_emotion(e) for e in val_labels])
    print('Validation samples preprocessed!')

    #build the vocabulary
    samples = train_samples + val_samples
    vocab = Vocabulary(' '.join(str(e) for e in samples))
    vocab.add_unkw()

    #TRAIN_DATA : transform the samples into tensors and pad them to the maximum length
    train_samples = [text_to_tensor(e.split(), vocab) for e in train_samples]
    #add a list of 0s to pad the others to max_len
    train_samples.append(torch.zeros(CONFIG['max_len']))
    train_samples = pad_sequence(train_samples, batch_first=True)
    print('Train samples padded!')
    #TRAIN_DATA : set the maximum length of an input to max_len
    train_samples = train_samples[:, :CONFIG['max_len']]

    #remove the added element
    train_samples = train_samples[:train_samples.shape[0]-1]

    #TRAIN_DATA : build the training data loader
    train_data_loader = build_loader(train_samples, train_labels, 'train data', CONFIG)

    #VALIDATION_DATA : transform the samples into tensors and pad them to the maximum length
    val_samples = [text_to_tensor(e.split(), vocab) for e in val_samples]
    val_samples.append(torch.zeros(CONFIG['max_len']))
    val_samples = pad_sequence(val_samples, batch_first=True)
    print('Validation samples padded!')
    #VALIDATION_DATA : set the maximum length of an input to max_len
    val_samples = val_samples[:, :CONFIG['max_len']]

    val_samples = val_samples[:val_samples.shape[0]-1]

    #VALIDATION_DATA : build the training data loader
    val_data_loader = build_loader(val_samples, val_labels, 'validation data', CONFIG)

    return vocab, train_data_loader, val_data_loader

def load_test(test_samples, test_labels, vocab, CONFIG):

    #transform the dataframe into (sample, label) pairs
    #for the moment samples is a numpy array
    samples = [preprocess(e) for e in test_samples.values]
    #build the labels tensor
    labels = torch.tensor([find_emotion(e) for e in test_labels])

    #transform the samples into tensors and pad them to the maximum length
    samples = [text_to_tensor(e.split(), vocab) for e in samples]
    #add a list of 0s to pad the others to max_len
    samples.append(torch.zeros(CONFIG['max_len']))
    samples = pad_sequence(samples, batch_first=True)
    print('Test data padded!')
    samples = samples[:, :CONFIG['max_len']]

    #remove the added list
    samples = samples[:samples.shape[0]-1]

    test_data_loader = build_loader(samples, labels, 'test data', CONFIG)
    return test_data_loader

def load_data(CONFIG=None):

    data = pd.read_csv(constants.DATA_PATH)

    samples = data['text']
    labels = data['emotions']

    #get train&validation sizes
    train_size = (int)(samples.shape[0] * CONFIG['train_cutoff'])
    val_size = (int)(samples.shape[0] * CONFIG['validation_cutoff'])
    
    #build samples tensors
    train_samples = samples[:train_size]
    validation_samples = samples[train_size:(train_size + val_size)]
    test_samples = samples[(train_size + val_size):]

    #build labels tensors
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:(train_size + val_size)]
    test_labels = labels[(train_size + val_size):]

    #build the data loaders
    #build train&validation in the same function to append the samples and create the vocabulary
    vocab, train_data_loader, validation_data_loader = load_train_val(train_samples, train_labels, 
                                                                      validation_samples, validation_labels, CONFIG)
    #won't add the test words in vocabulary
    #for unknown words use special word UNKW
    test_data_loader = load_test(test_samples, test_labels, vocab, CONFIG)
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
    torch.save(model, log_path + '/my_model.pth')

    #copy the used config
    logging.info('Saving the model s config at {}'.format(log_path + '/config.json'))
    copyfile('config.json', log_path + '/config.json')

def confusion_matrix(predicted, label, no_of_classes):

    conf_matrix = torch.zeros(no_of_classes, no_of_classes)
    for p, l in zip(predicted, label):
        conf_matrix[l, p] += 1

    return conf_matrix