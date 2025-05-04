import numpy as np

class Layer:
    def __init__(self, n_in: int, n_out: int, activation: str = 'relu'):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        limit = 1 / np.sqrt(n_in)
        # weights as nested lists: shape n_out x n_in
        self.W = [[np.random.uniform(-limit, limit) for _ in range(n_in)]
                  for _ in range(n_out)]
        # biases as list length n_out
        self.b = [0.0 for _ in range(n_out)]
        # caches
        self.A_prev = []
        self.Z = []
        self.dW = []
        self.db = []

    def activate(self, z: float) -> float:
        if self.activation == 'relu': return max(0.0, z)
        if self.activation == 'sigmoid': return 1.0 / (1.0 + np.exp(-z))
        return z

    def activate_deriv(self, z: float) -> float:
        if self.activation == 'relu': return 1.0 if z > 0 else 0.0
        if self.activation == 'sigmoid':
            s = 1.0/(1.0+np.exp(-z))
            return s*(1-s)
        return 1.0

    def forward(self, A_prev: list[float]) -> list[float]:
        self.A_prev = A_prev[:]
        Z = []
        A = []
        for j in range(self.n_out):
            z = sum(self.W[j][k] * A_prev[k] for k in range(self.n_in)) + self.b[j]
            Z.append(z)
            A.append(self.activate(z))
        self.Z = Z
        return A
    
    def backward(self, dA: list[float]) -> list[float]:
        m = 1  # assuming single-sample gradients
        dZ = [dA[j] * self.activate_deriv(self.Z[j]) for j in range(self.n_out)]
        # gradients
        self.dW = [[dZ[j] * self.A_prev[k] for k in range(self.n_in)]
                  for j in range(self.n_out)]
        self.db = dZ[:]
        # propagate to previous layer
        dA_prev = [0.0]*self.n_in
        for i in range(self.n_in):
            for j in range(self.n_out):
                dA_prev[i] += self.W[j][i] * dZ[j]
        return dA_prev

    def update(self, lr: float):
        for j in range(self.n_out):
            for k in range(self.n_in):
                self.W[j][k] -= lr * self.dW[j][k]
            self.b[j] -= lr * self.db[j]