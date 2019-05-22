import torch
import logging
from tqdm import tqdm
import time
import os

os.chdir('/home/andrei/Desktop/Poorify/AI')

#in-house imports
from models.model import MyModel
from models.trainer import Trainer
from utils import load_config, load_data, load_logging, save_model
import constants

def main():

    torch.cuda.empty_cache()

    #load logging
    if constants.ENABLE_LOGGING == True:
        load_logging()
    else:
        logging.getLogger().disabled = True

    #load the config
    CONFIG = load_config(constants.CONFIG_PATH)

    #check if cuda is available & is enabled in config
    constants.USE_CUDA = CONFIG['use_cuda'] & torch.cuda.is_available()
    if constants.USE_CUDA == True:
        constants.DEVICE = 'cuda:0'
    else:
        constants.DEVICE = 'cpu'

    #get the data generators
    vocab, train_generator, validation_generator, test_generator = load_data(CONFIG)

    #use incremented vocabulary size because we have an extra character which isn't
    #in the dictionary : 0 - padding character
    model = MyModel(vocab_size=vocab.size()+1, embedding_size=CONFIG['embedding_size'], 
                    output_size=CONFIG['output_size'], batch_size=CONFIG['batch_size'],
                    max_len=CONFIG['max_len'], hidden_size=CONFIG['rnn_size'],vocab=vocab)
    model = model.to(constants.DEVICE)

    trainer = Trainer(model=model, vocab=vocab, train_generator=train_generator,
                      val_generator=validation_generator, epochs=CONFIG['epochs'],
                      batch_size=CONFIG['batch_size'], max_grad_norm=CONFIG['max_grad_norm'],
                      lr=CONFIG['learning_rate'], loss=CONFIG['loss'], optim=CONFIG['optimizer'],
                      train_verbose=CONFIG['train_verbose'], val_verbose=CONFIG['validation_verbose'])
    
    model = trainer.train()

    #test the model
    conf_matrix = model.predict(test_generator)

    if CONFIG['save_model'] == True:
        save_model(model, conf_matrix)

if __name__ == "__main__":
    try:
        main()
        logging.info('Training ended!\n')
    except KeyboardInterrupt:
        print('Keyboard interrupt')
        logging.warning('Keyboard interrupt!\n')
        #TODO
        #save_checkpoint()