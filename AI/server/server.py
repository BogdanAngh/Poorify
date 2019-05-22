from flask import Flask, request, jsonify
import torch
import lyricwikia

import sys
sys.path.append('/home/andrei/Desktop/Poorify/AI/')

from models.model import MyModel
from utils import preprocess, text_to_tensor
from vocabulary import Vocabulary

app = Flask(__name__)

#load model
#model_state = torch.load('my_model.pth')
#create model
#vocab = Vocabulary('a')
#model = MyModel(vocab=vocab,vocab_size=12903, embedding_size=256, 
#               output_size=4, batch_size=64,
#                max_len=64, hidden_size=128)
#load model's state into class instantiation
#model.load_state_dict(model_state)

model = torch.load('my_model.pth')

@app.route('/api',methods=['POST'])
def predict():

    #get the data from the POST request.
    data = request.get_json(force=True)['input']

    data = text_to_tensor(data.split(), model.vocab)
    #if tensor to long slice it to desired length
    data = data[:64]
    #create auxiliary tensor
    zeros = torch.zeros(64)
    #pad the tensor with corresponding number of 0s
    zeros[:data.shape[0]] = data

    #run the model's prediction function
    result = model.single_prediction(zeros.unsqueeze(0).long())

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5002, debug=True)