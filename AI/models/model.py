import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
import logging

#in-house imports
from layers.prenet import Prenet
from layers.convolution import Convolution
from layers.postnet import Postnet
from layers.recurrence import Recurrence
from utils import confusion_matrix, preprocess, idx_to_emotion, text_to_tensor
from visualizer import plot_confusion_matrix

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, max_len, vocab,
                 batch_size=64, out_channels=200):
        super().__init__()

        self.vocab = vocab

        # EMBEDDING PARAMETERS
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # CONVOLUTION PARAMETERS
        self.max_len = max_len
        self.out_channels = out_channels
        self.batch_size = batch_size

        # POSTNET PARAMETERS
        self.output_size = output_size

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,
                                      embedding_dim  = self.embedding_size)
        logging.info('Embedding layer created : {}'.format(self.embedding))
        
        self.convolution = Convolution(in_channels = self.max_len, out_channels = self.out_channels, max_len = self.max_len)
        logging.info('Convolutional layer created : {}'.format(self.convolution))

        self.postnet = Postnet(in_size = self.out_channels*3*self.embedding_size//2, out_size = self.output_size, dropout_rate=0.5)
        logging.info('Postnet layer created : {}'.format(self.postnet))

        logging.info('MyModel created!')

    def forward(self, x, is_training=True):

        # in  = batch_size x max_len 
        # out = batch_size x max_len x embedding_size
        x = self.embedding(x)

        # out = batch_size x out_channels
        x = self.convolution(x)

        x = x.view(x.shape[0], 1, -1)

        # out = batch_size x output_size
        x = self.postnet(x, is_training)
    
        return x.squeeze(1)

    def predict(self, test_generator):

        conf_matrix = torch.zeros(self.output_size, self.output_size)

        for idx, (sample, label) in enumerate(test_generator):
            print(idx , ' / ', len(test_generator))
            predicted = torch.argmax(F.softmax(self.forward(sample.cuda(), is_training=False), dim=1), dim=1)
            cm = confusion_matrix(predicted, label, self.output_size)
            conf_matrix += cm
        
        label_names = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
        plot_confusion_matrix(conf_matrix.numpy(), label_names)

    def input_example(self, sample):

        initial_sample = sample

        #preprocess our input
        sample = preprocess(sample)
        #cast it to a tensor
        sample = [text_to_tensor(sample.split(), self.vocab)]
        #append a max_len tensor so the original tensor can be padded too
        sample.append(torch.zeros(self.max_len))
        sample = pad_sequence(sample, batch_first=True)
        
        #cut the input to the max_len
        sample = sample[:, :self.max_len]
        #remove the added element
        sample = sample[:sample.shape[0]-1]

        result = self.forward(sample.cuda(), is_training=False).squeeze(0)
        #print(F.softmax(result, dim=0))
        result = torch.argmax(F.softmax(result, dim=0), dim=0)

        #print(initial_sample, ' : ', idx_to_emotion[result.item()])

        return idx_to_emotion[result.item()]