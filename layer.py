import numpy as np
import math
import random
class Layer:
    def __init__(self, n_in: int, n_out: int, activation: str = 'relu', lr: float = 0.01):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.lr = lr
        # Xavier init
        limit = math.sqrt(6 / (n_in + n_out))
        self.W = [[random.uniform(-limit, limit) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out
        # caches
        self.A_prev = []
        self.Z = []
        self.dW = []
        self.db = []

    def _activate(self, Z: list[float]) -> list[float]:
        if self.activation == 'relu':
            return [max(0.0, z) for z in Z]
        if self.activation == 'softmax':
            exps = [math.exp(z) for z in Z]
            s = sum(exps)
            return [e / s for e in exps]
        return Z  # other activations not used here

    def _activate_deriv(self, z: float) -> float:
        if self.activation == 'relu':
            return 1.0 if z > 0 else 0.0
        return 1.0

    def forward(self, A_prev: list[float]) -> list[float]:
        self.A_prev = A_prev[:]
        self.Z = [sum(self.W[j][k] * A_prev[k] for k in range(self.n_in)) + self.b[j]
                  for j in range(self.n_out)]
        A = self._activate(self.Z)
        return A

    def backward(self, dA: list[float], y_true: list[float] = None) -> list[float]:
        # For softmax+cross-entropy, dA already equals (A - y)
        if self.activation == 'softmax':
            dZ = dA[:]
        else:
            dZ = [dA[j] * self._activate_deriv(self.Z[j]) for j in range(self.n_out)]
        # gradients
        self.dW = [[dZ[j] * self.A_prev[k] for k in range(self.n_in)] for j in range(self.n_out)]
        self.db = dZ[:]
        # propagate to previous
        dA_prev = [0.0] * self.n_in
        for i in range(self.n_in):
            for j in range(self.n_out):
                dA_prev[i] += self.W[j][i] * dZ[j]
        # update params
        for j in range(self.n_out):
            for k in range(self.n_in):
                self.W[j][k] -= self.lr * self.dW[j][k]
            self.b[j] -= self.lr * self.db[j]
        return dA_prev