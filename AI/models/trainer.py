import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam, SGD
import logging
from tqdm import tqdm
import time

#in-house imports
from models.model import MyModel
from visualizer import plot_loss
from emotionClasification import Emotion, findEmotion
import constants

class Trainer():
    def __init__(self, model, vocab, train_generator, 
                 val_generator, epochs=10, batch_size=32, 
                 max_grad_norm=5.0, lr=0.001, loss = 'mse',
                 optim='adam', train_verbose=10, val_verbose=5):
        
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
        elif loss == 'cross-entropy':
            self.loss_fn = nn.CrossEntropyLoss()

        logging.info('Trainer created!')

    def train_epoch(self, epoch):
            
        epoch_loss = 0.0
        #use a 0s tensor as starting hidden state
        t = time.time()
        for idx, (sample, label) in enumerate(self.train_generator):
            
            if sample.shape[0] < self.batch_size:
                continue

            sample = sample.to(constants.DEVICE)
            label = label.to(constants.DEVICE)

            #reset the gradients
            self.optimizer.zero_grad()

            result = self.model(sample)
            batch_loss = self.loss_fn(result, label)
            
            epoch_loss += batch_loss.item()

            batch_loss.backward()

            #clip the gradients to avoid explosions
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), self.max_grad_norm)

            #update the parameters
            self.optimizer.step()

            predicted = torch.argmax(F.softmax(result, dim=1), dim=1)
            acc = torch.sum(predicted == label)

            if idx % self.train_verbose == 0:
                print('Epoch {} - batch {} / {} -  train loss : {} - acc : {}'.format(epoch, idx, self.train_generator_size, 
                                                                                      batch_loss.item(), acc.item() / self.batch_size))
        
        return epoch_loss / self.train_generator_size
        
    def val_epoch(self, epoch):
        
        #validation epoch won't do any updates
        with torch.no_grad():
            epoch_loss = 0.0

            for idx, (sample, label) in enumerate(self.val_generator):
                
                if sample.shape[0] < self.batch_size:
                    continue

                sample = sample.to(constants.DEVICE)
                label = label.to(constants.DEVICE)

                result = self.model(sample)
                batch_loss = self.loss_fn(result, label)
                epoch_loss += batch_loss.item()

                predicted = torch.tensor(torch.argmax(F.softmax(result, dim=1), dim=1))
                acc = torch.sum(predicted == label)

                if idx % self.val_verbose == 0:
                    print('Epoch {} - batch {} / {} - validation loss : {} - accuracy : {}'.format(epoch, idx, self.val_generator_size, \
                                                                                                   batch_loss.item(), acc.item() / self.batch_size))
            
            return epoch_loss / self.val_generator_size

    def train(self):

        logging.info('Starting the training...')

        train_loss = []
        val_loss = []
        for e in tqdm(range(self.epochs), ascii=True, desc='Epochs'):
            t_loss = self.train_epoch(e)
            v_loss = self.val_epoch(e)

            #keep the losses from each epoch
            train_loss.append(t_loss)
            val_loss.append(v_loss)

        plot_loss(train_loss, val_loss)


            