import numpy as np
from layer import Layer

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01):
        self.layers = []
        for i in range(1, len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], 'relu'))
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], 'sigmoid'))
        self.lr = learning_rate

    def forward(self, X: list[list[float]]) -> list[list[float]]:
        A = []
        for x in X:
            out = x[:]
            for layer in self.layers:
                out = layer.forward(out)
            A.append(out)
        return A

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100):
        for _ in range(epochs):
            for x, y in zip(X.tolist(), Y.tolist()):
                # forward & get output
                outs = [x]
                for layer in self.layers:
                    outs.append(layer.forward(outs[-1]))
                # compute initial gradient (MSE loss)
                dA = [(2*(outs[-1][j] - y[j])) for j in range(len(y))]
                # backward
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)
                # update
                for layer in self.layers:
                    layer.update(self.lr)

    def predict(self, X: np.ndarray) -> list[int]:
        preds = []
        for x in X.tolist():
            out = x[:]
            for layer in self.layers:
                out = layer.forward(out)
            preds.append(int(np.argmax(out)))
        return preds