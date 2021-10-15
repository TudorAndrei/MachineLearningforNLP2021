import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x: np.array) -> np.array:
        self.mask = np.random.rand(*x.shape) > self.p
        # Scale the mask to even out missing neurons
        x = x * self.mask / self.p
        return x

    def backward(self, grad: np.array = np.array([[1]])) -> np.array:
        # Scale the mask to even out missing neurons
        return grad * self.mask / self.p


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> np.array:
        return grad * (self.forward(x) * (1 - self.forward(x)))


class MeanSquaredError:
    def __init__(self):
        pass

    def forward(self, y_pred: np.array, y_true: np.array) -> float:
        return np.mean(0.5 * (y_true - y_pred) ** 2)

    def backward(self, y_pred: np.array, y_true: np.array, grad: np.array = np.array([[1]])) -> np.array:
        return grad * (y_pred - y_true)


class FullyConnectedLayer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.input_size, self.output_size)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, x: np.array) -> np.array:
        return np.matmul(x, self.weights) + self.bias

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> np.array:
        x_grad = np.matmul(grad, self.weights.T)
        W_grad = np.matmul(x.T, grad)
        b_grad = grad

        return (x_grad, W_grad, b_grad)


class NeuralNetwork:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: List[int],
                 activation=Sigmoid,
                 dropout: float = 0.5):
        s = [input_size] + hidden_sizes + [output_size]

        self.layers = [FullyConnectedLayer(
            s[i], s[i+1]) for i in range(len(s) - 1)]
        self.dropouts = [Dropout(dropout) for i in range(len(s) - 2)]
        self.activation = activation()

    def forward(self, x: np.array) -> None:
        self.layer_inputs = []
        self.activ_inputs = []

        for layer, dropout in zip(self.layers[:-1], self.dropouts):
            self.layer_inputs.append(x)
            x = layer.forward(x)
            self.activ_inputs.append(x)
            x = self.activation.forward(x)

            # Dropout Layer
            x = dropout.forward(x)

        # The last layer should not be using an activation function
        self.layer_inputs.append(x)
        x = self.layers[-1].forward(x)

        return x

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) -> Tuple[np.array]:
        W_grads = []
        b_grads = []

        grad, W_grad, b_grad = self.layers[-1].backward(
            self.layer_inputs[-1], grad)
        W_grads.append(W_grad)
        b_grads.append(b_grad)

        for i in reversed(range(len(self.activ_inputs))):
            # Dropout Layer
            grad = self.dropouts[i].backward(grad)

            grad = self.activation.backward(self.activ_inputs[i], grad)
            grad, W_grad, b_grad = self.layers[i].backward(
                self.layer_inputs[i], grad)
            W_grads.append(W_grad)
            b_grads.append(b_grad)

        return grad, list(reversed(W_grads)), list(reversed(b_grads))



class SGD:
    def __init__(self, nn:object, lr:float):
        self.nn = nn
        self.lr = lr

    def update(self, W_grads: List[np.array], b_grads: List[np.array]) -> None:
        for i in range(len(self.nn.layers)):
            self.nn.layers[i].weights -= self.lr * W_grads[i]
            self.nn.layers[i].bias -= self.lr * b_grads[i]


class Momentum:
    def __init__(self, nn: object, lr: float, mu: float):
        self.v_W = [0] * len(nn.layers)
        self.v_b = [0] * len(nn.layers)

        self.nn = nn
        self.lr = lr
        self.mu = mu

    def update(self, W_grads: List[np.array], b_grads: List[np.array]) -> None:
        for i in range(len(self.nn.layers)):
            self.v_W[i] = self.mu * self.v_W[i] - self.lr * W_grads[i]
            self.v_b[i] = self.mu * self.v_b[i] - self.lr * b_grads[i]

            self.nn.layers[i].weights += self.v_W[i]
            self.nn.layers[i].bias += self.v_b[i]


class Adam:
    def __init__(self, nn: object, lr: float, beta1: float, beta2: float):
        self.m_W = [0] * len(nn.layers)
        self.m_b = [0] * len(nn.layers)

        self.v_W = [0] * len(nn.layers)
        self.v_b = [0] * len(nn.layers)

        self.nn = nn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def update(self, W_grads: List[np.array], b_grads: List[np.array]) -> None:
        for i in range(len(self.nn.layers)):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * W_grads[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * b_grads[i]

            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * W_grads[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * b_grads[i]**2

            self.nn.layers[i].weights -= self.lr * self.m_W[i] / (np.sqrt(self.v_W[i]) + 1e-8)
            self.nn.layers[i].bias -= self.lr * self.v_b[i] / (np.sqrt(self.v_b[i]) + 1e-8)



if __name__ == "__main__":
    # Network Initialization (with Dropout)
    net = NeuralNetwork(2, 1, [2], Sigmoid, dropout=0.5)

    # Setting the layer weights for reproduction purposes
    net.layers[0].weights = np.array([[0.5, 0.75], [0.25, 0.25]])
    net.layers[1].weights = np.array([[0.5], [0.5]])

    # Loss
    loss_function = MeanSquaredError()

    # Optimizer
    optimizer = SGD(net, lr=0.1)
    # optimizer = Momentum(net, lr=0.1, mu=0.0)
    # optimizer = Adam(net, 0.001, 0.9, 0.999)

    # XOR Dataset
    inputs = [
        (np.array([[0, 0]]), np.array([0])),
        (np.array([[0, 1]]), np.array([1])),
        (np.array([[1, 0]]), np.array([1])),
        (np.array([[0, 0]]), np.array([0]))
    ]

    epochs = 50000

    # tqdm (you need to install it using `pip3 install tqdm`)
    # will show a progress bar for the loop
    for epoch in tqdm(range(epochs)):
        # It is always a good idea to shuffle the dataset
        np.random.shuffle(inputs)
        
        for x, y in inputs:
            # Forward Pass
            pred = net.forward(x)

            # Loss Calculation
            loss = loss_function.forward(pred, y)

            # Backward Pass
            grad = loss_function.backward(pred, y)
            grad, W_grads, b_grads = net.backward(x, grad)

            optimizer.update(W_grads, b_grads)


    # Test that the network has learned something
    for x, y in inputs:
        # Forward Pass
        pred = net.forward(x)

        print(f"Input: {x}, Output: {pred}, Desired Output: {y}")
