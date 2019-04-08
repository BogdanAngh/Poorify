#in-house imports
from models.model import MyModel
from models.trainer import Trainer
from utils import load_config, load_data
import constants

import torch

def main():

    #load the config
    CONFIG = load_config(constants.CONFIG_PATH)

    #check if cuda is available & is enabled in config
    constants.USE_CUDA = CONFIG['use_cuda'] & torch.cuda.is_available()
    if constants.USE_CUDA == True:
        constants.DEVICE = 'cuda:0'
    else:
        constants.DEVICE = 'cpu'

    #get the data generators
    vocab, train_generator, validation_generator = load_data(constants.DATASET_PATH, CONFIG)

    model = MyModel(vocab.size(), CONFIG['embedding_size'], 
                    CONFIG['rnn_size'], CONFIG['output_size'])

    print(model)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt')
        #TODO
        #save_checkpoint()