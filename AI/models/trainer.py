import torch.nn as nn
import torch
from torch.optim import Adam, SGD
import logging
from tqdm import tqdm

#in-house imports
from models.model import MyModel
from visualizer import plot_loss

class Trainer():
    def __init__(self, model, vocab, train_generator, 
                 val_generator, epochs=10, batch_size=32, 
                 max_grad_norm=5.0, lr=0.001, loss = 'mse',
                 optim='adam', train_verbose=10, val_verbose=10):
        
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.vocab = vocab

        self.train_generator = train_generator
        self.val_generator = val_generator

        self.train_generator_size = len(train_generator)
        self.val_generator_size = len(val_generator)

        self.train_verbose = train_verbose
        self.val_verbose = val_verbose

        if optim == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        elif optim == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr = self.lr)

        if loss == 'mse':
            self.loss_fn = nn.MSELoss()

        logging.info('Trainer created!')

    def train_epoch(self, epoch):
            
        epoch_loss = 0.0
        #use a 0s tensor as starting hidden state
        prev_hidden = torch.zeros(self.batch_size, self.model.rnn_size)
        for idx, (sample, label) in enumerate(self.train_generator):
            
            if sample.shape[0] < self.batch_size:
                continue

            #reset the gradients
            self.optimizer.zero_grad()

            hidden_state = self.model(sample, prev_hidden)
            logits = self.model.get_logits(hidden_state)
            batch_loss = self.loss_fn(logits, label)
            
            epoch_loss += batch_loss.item()

            batch_loss.backward()

            #clip the gradients to avoid explosions
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), self.max_grad_norm)

            #update the parameters
            self.optimizer.step()

            #use generated hidden state as the next input hidden states
            hidden_state.detach_()
            prev_hidden = hidden_state

            if idx % self.train_verbose:
                print('Epoch {} - train loss : {}'.format(epoch, batch_loss.item()))
        
        return epoch_loss / self.train_generator_size
        
    def val_epoch(self, epoch, hidden_start=None):
        
        #validation epoch won't do any updates
        with torch.no_grad():
            epoch_loss = 0.0
            prev_hidden = torch.zeros(self.batch_size, self.model.rnn_size)

            for idx, (sample, label) in enumerate(self.val_generator):
                
                if sample.shape[0] < self.batch_size:
                    continue

                hidden_state = self.model(sample, prev_hidden)
                logits = self.model.get_logits(hidden_state)
                batch_loss = self.loss_fn(logits, label)
                epoch_loss += batch_loss.item()

                prev_hidden = hidden_state

                if idx % self.val_verbose:
                    print('Epoch {} - validation loss : {}'.format(epoch, batch_loss.item()))
            
            return epoch_loss / self.val_generator_size

    def train(self):

        logging.info('Starting the training...')

        train_loss = []
        val_loss = []
        hidden_start = torch.zeros(self.batch_size, self.model.rnn_size)
        for e in tqdm(range(self.epochs), ascii=True, desc='Epochs'):
            t_loss = self.train_epoch(e)
            v_loss = self.val_epoch(e, hidden_start)

            #keep the losses from each epoch
            train_loss.append(t_loss)
            val_loss.append(v_loss)

        plot_loss(train_loss, val_loss)


            