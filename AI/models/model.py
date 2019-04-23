import torch.nn as nn
import torch
import logging

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_size, output_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.output_size = output_size

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,
                                      embedding_dim  = self.embedding_size)
        logging.info('Embedding layer created : {}'.format(self.embedding))

        self.rnn = nn.LSTM(self.embedding_size,
                          self.rnn_size,
                          1)
        logging.info('Recurrent layer created : {}'.format(self.rnn))

        self.logits = nn.Linear(in_features  = self.rnn_size,
                                out_features = self.output_size)
        logging.info('Linear layer created : {}'.format(self.logits))

        self.logits_activation = nn.LeakyReLU(1)

        logging.info('MyModel created!')

    #multiply the result by 2 so we get results in the range (-3, 3)
    def get_logits(self, hidden_states, temperature=1.0):
        return self.logits_activation(self.logits(hidden_states) / temperature)

    def forward(self, x, hidden_start=None):

        in_len = x.shape[0]

        #returns a batch_size x in_len x embedding_size
        x = self.embedding(x)

        #use hidden_start as the first hidden input(usually tensor of 0s)
        prev_hidden = hidden_start
        for t in range(in_len):
            #feed the gru cell with the input for the t-th time step
            out, hidden_state = self.rnn(x[t, :, :], prev_hidden)
            #use the generated hidden state for the next time step
            prev_hidden = hidden_state

        return (hidden_state, out)
