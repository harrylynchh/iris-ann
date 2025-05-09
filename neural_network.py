import numpy as np
from layer import Layer
import random

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], lr: float = 0.01):
        self.layers = []
        self.lr = lr
        # hidden layers with ReLU
        for i in range(1, len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], 'relu', lr))
        # output layer with softmax
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], 'softmax', lr))

    def train(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, epochs: int = 100) -> None:
        n = X.shape[0]
        for epoch in range(1, epochs+1):
            idxs = list(range(n))
            random.shuffle(idxs)
            for j in idxs:
                out = X[j].tolist()
                # forward pass
                for layer in self.layers:
                    out = layer.forward(out)
                # compute gradient for cross-entropy: dA = A - y_true
                y_true = Y[j].tolist()
                dA = [out[k] - y_true[k] for k in range(len(out))]
                # backward pass
                for layer in reversed(self.layers):
                    out_grad = layer.backward(dA)
                    dA = out_grad
            # validation accuracy
            preds = self.predict(X_val)
            acc = sum(int(p==t) for p,t in zip(preds, np.argmax(Y_val, axis=1))) / len(preds)
            print(f"Epoch {epoch}/{epochs} - Val Acc: {acc:.4f}")

    def predict(self, X: np.ndarray) -> list[int]:
        preds = []
        for x in X.tolist():
            out = x[:]
            for layer in self.layers:
                out = layer.forward(out)
            preds.append(int(out.index(max(out))))
        return preds