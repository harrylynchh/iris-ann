import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math, random
from preprocessor import load_and_preprocess

# --- Neural Network Classes without matrix ops ---
class Layer:
    def __init__(self, n_in: int, n_out: int, activation: str, lr: float, momentum: float):
        self.n_in, self.n_out = n_in, n_out
        self.activation = activation
        self.lr = lr
        self.momentum = momentum
        limit = 1 / math.sqrt(n_in)  # slide heuristic
        self.W = [[random.uniform(-limit, limit) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out
        # cache
        self.A_prev: list[float] = []
        self.Z: list[float] = []
        # grads & momentum buffers
        self.dW = [[0.0] * n_in for _ in range(n_out)]
        self.db = [0.0] * n_out
        self.vW = [[0.0] * n_in for _ in range(n_out)]
        self.vb = [0.0] * n_out

    def _activate(self, Z: list[float]) -> list[float]:
        if self.activation == 'relu':
            return [max(0.0, z) for z in Z]
        if self.activation == 'softmax':
            exps = [math.exp(z) for z in Z]
            s = sum(exps)
            return [e / s for e in exps]
        return Z

    def _deriv(self, z: float) -> float:
        if self.activation == 'relu':
            return 1.0 if z > 0 else 0.0
        return 1.0  # other cases handled outside

    def forward(self, A_prev: list[float]) -> list[float]:
        self.A_prev = A_prev[:]
        # Compute Z = W · A_prev + b
        self.Z = [sum(self.W[j][k] * A_prev[k] for k in range(self.n_in)) + self.b[j]
                  for j in range(self.n_out)]
        # Apply activation
        return self._activate(self.Z)

    def backward(self, dA: list[float]) -> list[float]:
        """
        Backward pass: compute gradients and update parameters via SGD.
        Returns gradient w.r.t. previous layer activation.
        """
        # 1) compute dZ
        if self.activation == 'softmax':
            dZ = dA[:]
        else:
            dZ = [dA[j] * self._deriv(self.Z[j]) for j in range(self.n_out)]
        # 2) compute gradients
        self.dW = [[dZ[j] * self.A_prev[k] for k in range(self.n_in)] for j in range(self.n_out)]
        self.db = dZ[:]
        # 3) compute dA_prev
        dA_prev = [0.0] * self.n_in
        for i in range(self.n_in):
            for j in range(self.n_out):
                dA_prev[i] += self.W[j][i] * dZ[j]
        # 4) update parameters
        for j in range(self.n_out):
            for k in range(self.n_in):
                self.W[j][k] -= self.lr * self.dW[j][k]
            self.b[j] -= self.lr * self.db[j]
        return dA_prev


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], lr: float = 0.01, momentum: float = 0.9):
        self.layers = []
        self.lr, self.momentum = lr, momentum
        # hidden layers
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i], 'relu', lr, momentum))
        # output layer
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], 'softmax', lr, momentum))

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 16
    ) -> None:
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            idxs = list(range(n))
            random.shuffle(idxs)
            # process in minibatches
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idxs = idxs[start:end]
                # aggregate gradients over the batch
                # initialize dA for each sample sequentially
                for j in batch_idxs:
                    out = X[j].tolist()
                    for layer in self.layers:
                        out = layer.forward(out)
                    y_true = Y[j].tolist()
                    dA = [out[k] - y_true[k] for k in range(len(out))]
                    for layer in reversed(self.layers):
                        dA = layer.backward(dA)
            # validation accuracy
            preds = self.predict(X_val)
            acc = sum(p == t for p, t in zip(preds, np.argmax(Y_val, axis=1))) / len(preds)
            print(f"Epoch {epoch}/{epochs} - Val Acc: {acc:.4f}")

    def predict(self, X: np.ndarray) -> list[int]:
        preds = []
        for x in X.tolist():
            out = x[:]
            for layer in self.layers:
                out = layer.forward(out)
            preds.append(int(max(range(len(out)), key=lambda i: out[i])))
        return preds


# --- Main Execution ---
if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_preprocess('iris_dataset.txt')
    nn = NeuralNetwork([4, 8, 6, 3], lr=0.005, momentum=0.9)
    nn.train(X_train, Y_train, X_val, Y_val, epochs=100)
    preds = nn.predict(X_test)
    acc = sum(p == t for p, t in zip(preds, np.argmax(Y_test, axis=1))) / len(preds)
    print(f"Test Accuracy: {acc:.4f}")
