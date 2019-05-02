import torch.nn as nn
import torch
import logging

#in-house imports
from layers.prenet import Prenet
from layers.convolution import Convolution
from layers.postnet import Postnet
from layers.recurrence import Recurrence

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size,
                 max_len, batch_size=64, out_channels=100, kernel_size=5, 
                 hidden_size=256):
        super().__init__()

        # EMBEDDING PARAMETERS
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # CONVOLUTION PARAMETERS
        self.kernel_size = kernel_size
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
        
        self.convolution = Convolution(in_channels = self.max_len, out_channels = self.out_channels,
                                       kernel_size = self.kernel_size, max_len = self.max_len)
        logging.info('Convolutional layer created : {}'.format(self.convolution))

        # we apply pooling on each output channels of the convolutional layer 
        # so each channels gets reducted to a few features depending on the pooling kernel size
        feature_size = self.max_len // (self.max_len - self.kernel_size + 1)
        #self.recurrence_input = self.out_channels * feature_size
        self.recurrence_input = self.embedding_size * 300
        self.recurrence = Recurrence(input_size = self.recurrence_input, hidden_size = self.hidden_size)
        logging.info('Recurrent layer created : {}'.format(self.recurrence))

        self.postnet = Postnet(in_size = self.hidden_size, out_size = self.output_size)
        logging.info('Postnet layer created : {}'.format(self.postnet))

        logging.info('MyModel created!')

    def forward(self, x, is_training=True):

        # in  = batch_size x max_len 
        # out = batch_size x max_len x embedding_size
        x = self.embedding(x)

        #x = self.prenet(x)
        
        # out = batch_size x out_channels
        x = self.convolution(x)

        # out = batch_size x hidden_size
        x = x.view(x.shape[0], 1, -1)
        x = self.recurrence(x)

        # out = batch_size x output_size
        x = self.postnet(x, is_training)
    
        return x.squeeze(1)