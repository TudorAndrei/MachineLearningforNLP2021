import numpy as np
from typing import List, Tuple


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> np.array:
        # don't mind me this week
        pass


class MeanSquaredError:
    def __init__(self):
        pass

    def forward(self, y_pred: np.array, y_true: np.array) -> float:
        return np.mean(0.5 * (y_true - y_pred) ** 2)

    def backward(self, y_pred: np.array, y_true: np.array, grad: np.array = np.array([[1]])) -> np.array:
        # don't mind me this week
        pass


class FullyConnectedLayer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.input_size, self.output_size)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, x: np.array) -> np.array:
        return np.matmul(x, self.weights) + self.bias

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> np.array:
        # don't mind me this week
        pass


class NeuralNetwork:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: List[int],
                 activation=Sigmoid):
        s = [input_size] + hidden_sizes + [output_size]

        self.layers = [FullyConnectedLayer(
            s[i], s[i+1]) for i in range(len(s) - 1)]
        self.activation = activation()

    def forward(self, x: np.array) -> None:
        for layer in self.layers[:-1]:
            x = layer.forward(x)
            x = self.activation.forward(x)

        # The last layer should not be using an activation function
        x = self.layers[-1].forward(x)

        return x

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> Tuple[np.array]:
        # don't mind me this week
        pass


if __name__ == "__main__":
    # Network Initialization
    net = NeuralNetwork(2, 1, [2], Sigmoid)

    # Setting the layer weights
    net.layers[0].weights = np.array([[0.5, 0.75], [0.25, 0.25]])
    net.layers[1].weights = np.array([[0.5], [0.5]])

    # Loss
    loss_function = MeanSquaredError()

    # Input
    x = np.array([[1, 1]])
    y = np.array([[0]])

    # Forward Pass
    pred = net.forward(x)

    # Loss Calculation
    loss = loss_function.forward(pred, y)

    print(f"Prediction: {pred}")
    print(f"Loss: {loss}")
