'''
neural_network.py
5/9/2025
Harry Lynch
Implementation of the NeuralNetwork class which constructs a sequence of Layer
objects given an array of layer sizes.  Facilitates forward and backward propagation
through the layers in the network in a training algorithm which features learning rate
decay by a factor of 0.5 every 25 epochs.  Prints validation accuracy after each epoch
and is able to take in test inputs w/ the predict function.  
'''
import numpy as np
from layer import Layer
import random
import math

class NeuralNetwork:
    '''
    __init__
    Construct the network making a layer of size layer_sizes[i] w/ hidden layer
    using RELU and output layer using sigmoid.
    '''
    def __init__(self, layer_sizes: list[int], lr: float = 0.1):
        self.layers: list[Layer] = []
        self.lr: float = lr
        # hidden layers use ReLU for activation
        for i in range(1, len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], 'relu', lr))
        # output layer with sigmoid for activation (gives prob-like value btwn 0 & 1)
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], 'sigmoid', lr))

    '''
    train
    Given a number of epochs, shuffle the order in which the test set is introduced to the network
    and run forward/backward propagation over each Layer using Root Mean Square Error 
    as the loss function.  After each epoch of training, try against the validation set and report accuracy.
    '''
    def train(self, X: np.ndarray, Y: np.ndarray,
              X_val: np.ndarray, Y_val: np.ndarray,
              epochs: int = 100) -> None:
        # Num samples in the set
        n = X.shape[0]
        for epoch in range(1, epochs+1):
            # Learning rate decay (halve every 25)
            if epoch % 25 == 0:
                self.lr *= 0.5

            # Shuffle indices each epoch-- got better training this way
            idxs = list(range(n))
            random.shuffle(idxs)

            for j in idxs:
                out = X[j].tolist()
                # forward pass
                for layer in self.layers:
                    out = layer.forward(out)

                # Extract labels
                y_true = Y[j].tolist()
                # Num of output neurons
                O = len(out)
                # Given the final activations in the output layer, calculate error
                mse = sum((out[k] - y_true[k])**2 for k in range(O)) / O
                # Use an epsilon if result of mse is 0
                rmse = math.sqrt(mse) if mse > 0 else 1e-8
                # derivative of RMSE with respect to each output neuron
                dA = [(out[k] - y_true[k]) / (O * rmse) for k in range(O)]

                # Beginning with the above, pass the error through each layer in reverse-order
                for layer in reversed(self.layers):
                    dA = layer.backward(dA, self.lr)

            # Calculate validation accuracy
            preds = self.predict(X_val)
            acc = sum(int(p==t) for p,t in zip(preds, np.argmax(Y_val, axis=1))) / len(preds)
            print(f"Epoch {epoch}/{epochs} - Val Acc: {acc:.4f}")

    '''
    predict
    Given a set of samples each containing the four features, run them thru forward
    propagation and classify each sample.  Returns a list of 1-hot list indices
    '''
    def predict(self, X: np.ndarray) -> list[int]:
        preds = []
        for x in X.tolist():
            out = x[:]
            # Run the sample thru the network
            for layer in self.layers:
                out = layer.forward(out)
            # pick the index of the largest activation
            preds.append(int(np.argmax(out)))
        # Return a prediction for each sample passed as a list of indices
        return preds