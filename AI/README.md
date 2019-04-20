Use ```pip install -r requirements.txt``` to install all the requirements.<br>
If you are running this in an anaconda environment use ```conda install --file requirements.txt```(recommended because it will also install all the dependencies).

#What's new:

- using words embedding instead of character embedding
- added new types of layers
- added more modularity over the code and over the addition of a new layer

#Arhitecture
1. Prenet : applies a bottleneck network(usually 256 -> 128) over the embedded input.
2. Convolution : applies 3 convolutions of different kernel size over the input in order to get k-grams. Use padding to keep the shapes the same after each convolution. Applies a max-pooling of kernel size 2 at the end.
3. Recurrence : for the moment just a bidirectional GRU to get information in both directions.
4. Postnet : output layer which applies a tanh over the recurrence's output. Uses dropout to reduce overfitting. After appling the activation function we multiply by 3 to get our results in the right range.

#Accuracy

Quite bad for the moment. Must experiment more with the hyperparameters and use more metrics like AUC and confusion matrix.
 

