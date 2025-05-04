import numpy as np
from preprocessor import load_and_preprocess
from neural_network import NeuralNetwork

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_preprocess("iris_dataset.txt")
nn = NeuralNetwork([4, 8, 3], learning_rate=0.01)
nn.train(X_train, Y_train, epochs=500)
preds = nn.predict(X_test)
acc = sum(1 for p,t in zip(preds, np.argmax(Y_test,axis=1)) if p==t) / len(preds)
print(f'Test Accuracy: {acc:.4f}')