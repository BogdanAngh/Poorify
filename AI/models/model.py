import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

#in-house imports
from layers.prenet import Prenet
from layers.convolution import Convolution
from layers.postnet import Postnet
from layers.recurrence import Recurrence
from utils import confusion_matrix
from visualizer import plot_confusion_matrix

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, max_len, 
                 batch_size=64, out_channels=100, hidden_size=256):
        super().__init__()

        # EMBEDDING PARAMETERS
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # CONVOLUTION PARAMETERS
        self.max_len = max_len
        self.out_channels = out_channels

        # RECURRENCE PARAMETERS 
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # POSTNET PARAMETERS
        self.output_size = output_size


        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,
                                      embedding_dim  = self.embedding_size)
        logging.info('Embedding layer created : {}'.format(self.embedding))
        
        #self.prenet = Prenet()
        
        self.convolution = Convolution(in_channels = self.max_len, out_channels = self.out_channels, max_len = self.max_len)
        logging.info('Convolutional layer created : {}'.format(self.convolution))

        # we apply pooling on each output channels of the convolutional layer 
        # so each channels gets reducted to a few features depending on the pooling kernel size
        #self.recurrence_input = self.embedding_size
        #self.recurrence = Recurrence(input_size = self.recurrence_input, hidden_size = self.hidden_size)
        #logging.info('Recurrent layer created : {}'.format(self.recurrence))

        self.postnet = Postnet(in_size = self.out_channels*3*self.embedding_size//2, out_size = self.output_size, dropout_rate=0.2)
        logging.info('Postnet layer created : {}'.format(self.postnet))

        logging.info('MyModel created!')

    def forward(self, x, is_training=True):

        # in  = batch_size x max_len 
        # out = batch_size x max_len x embedding_size
        x = self.embedding(x)

        #x = x.view(32*128,-1)
        #x = self.prenet(x)
        #x = x.view(32, 128, -1)

        # out = batch_size x out_channels
        x = self.convolution(x)

        # out = batch_size x hidden_size
        x = x.view(x.shape[0], 1, -1)
        #x = self.recurrence(x)

        # out = batch_size x output_size
        x = self.postnet(x, is_training)
    
        return x.squeeze(1)

    def predict(self, test_generator):

        conf_matrix = torch.zeros(self.output_size, self.output_size)

        for sample, label in test_generator:
            predicted = torch.argmax(F.softmax(self.forward(sample.cuda(), is_training=False), dim=1), dim=1)
            cm = confusion_matrix(predicted, label, self.output_size)
            conf_matrix += cm
        
        label_names = ['Happy', 'Angry', 'Sad', 'Calm']
        plot_confusion_matrix(conf_matrix.numpy(), label_names)
