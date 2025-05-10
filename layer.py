'''
layer.py
5/9/2025
Harry Lynch
Implementation of the Layer class which holds the weights and biases for each neuron
aswell as the number of inputs taken from the previous Layer in the Network.
Caches key gradients for use in the back propagation algorithm and supports
RELU and Sigmoid activation functions.  Note that  RELU is used in hidden layers
and Sigmoid is preferred for output as it produces a probability-like value btwn 0 and 1 
'''
import math
import random
class Layer:
    '''
    __init__
    Initialize all the members of the Layer class (see comments for item-by-item explanation)
    '''
    def __init__(self, n_in: int, n_out: int, activation: str = 'relu', lr: float = 0.1):
        # Number of inputs coming into the layer
        self.n_in = n_in
        # Number of neurons in the layer
        self.n_out = n_out
        # Activation function associated with the layer (I chose relu (sigmoid for output layer))
        self.activation = activation
        # Learning rate
        self.lr = lr
        # Set weights by +/- numWeights
        limit = 1.0 / n_in
        self.W = [[random.uniform(-limit, limit) for _ in range(n_in)] for _ in range(n_out)]
        # Start biases at 0
        self.b = [0.0] * n_out
        
        # THE FOLLOWING ARE FOR BACKPROP:
        # Previous layer's initial input
        self.A_prev = []
        # Potentials
        self.Z = []
        # Derivatives of weights and biases 
        self.dW = []
        self.db = []

    '''
    activate
    returns the activation values using a requested activation function given
    a list of potentials Z. 
    '''
    def activate(self, Z: list[float]) -> list[float]:
        if self.activation == 'relu':
            return [max(0.0, z) for z in Z]
        if self.activation == 'sigmoid':
            return [1.0 / (1.0 + math.exp(-z)) for z in Z]
        # ONLY SUPPORT RELU AND SIGMOID
        print("ERROR: Layer with activation other than relu or sigmoid")
        return Z

    '''
    activate_deriv
    return the derivative of ONE potential, z per the current layer's assigned
    activation function
    '''
    def activate_deriv(self, z: float) -> float:
        if self.activation == 'relu':
            return 1.0 if z > 0 else 0.0
        
        elif self.activation == 'sigmoid':
            # Deriv of sigmoid = sigmoid(p) * (1 - sigmoid(p))
            sig = 1.0/(1.0+math.exp(-z))
            return sig * (1.0 - sig)
        print("ERROR: Layer with activation other than relu or sigmoid")
    
    '''
    forward
    Compute potentials given the previous layer's activations, activate and then
    return forward to the next layer (see NeuralNetworks forward implementation)
    '''
    def forward(self, A_prev: list[float]) -> list[float]:
        # Set prevs to the passed values for backprop later
        self.A_prev = A_prev[:]
        # Calculate the potentials foreach neuron given all the activations from
        # the previous layers
        self.Z = [sum(self.W[j][k] * A_prev[k] for k in range(self.n_in)) + self.b[j]
                  for j in range(self.n_out)]
        # return a list of activations to the next layer
        A = self.activate(self.Z)
        return A

    '''
    backward
    Execute backward propagation on a singular layer of the network, taking in the
    gradients of the upstream neurons.  Ultimately this attempts to minimize error
    using gradient descent adjusting weights and biases based off each neuron's
    contribution to the total error.
    '''
    def backward(self, dA: list[float], curr_lr: float) -> list[float]:
        # Since I implemented lr decay, update on each call of backprop to ensure
        # it's current
        self.lr = curr_lr
        # Calculate initial gradients from prior activation derivatives foreach neuron in layer
        dZ = [dA[j] * self.activate_deriv(self.Z[j]) for j in range(self.n_out)]
        # Gradients of weight foreach input foreach neuron in the current layer
        self.dW = [[dZ[j] * self.A_prev[k] for k in range(self.n_in)] for j in range(self.n_out)]
        # Gradients of each neuron's bias is equal to the respective initial gradient w/ respect to the previous activation
        self.db = dZ[:]
        # Propogate to the previous layer starting each gradient at 0
        dA_prev = [0.0] * self.n_in
        # Foreach input into each neuron in the layer, calculate the propogated error to pass back
        for i in range(self.n_in):
            for j in range(self.n_out):
                dA_prev[i] += self.W[j][i] * dZ[j]
        # Using the error propogated back, adjust the weights and biases of each neuron
        for j in range(self.n_out):
            for k in range(self.n_in):
                self.W[j][k] -= self.lr * self.dW[j][k]
            self.b[j] -= self.lr * self.db[j]
        # Return this layer's gradients to be passed back
        return dA_prev