import numpy as np
from random import random

# save activation and derivative
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some prediction


class MLP(object):

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random weight
        weight = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weight.append(w)
        self.weight = weight

        activation = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activation.append(a)
        self.activation = activation

        derivative = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivative.append(d)
        self.derivative = derivative

    def forward_propogate(self, inputs):

        activation = inputs
        self.activation[0] = activation

        for i, w in enumerate(self.weight):
            # calculate net inputs
            net_inputs = np.dot(activation, w)

            # calculate the activation
            activation = self._sigmoid(net_inputs)
            self.activation[i+1] = activation

        return activation

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivative))):
             # get activation for previous layer
            activation = self.activation[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activation)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activation = self.activation[i]

            # reshape activations as to have them as a 2d column matrix
            current_activation = current_activation.reshape(current_activation.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivative[i] = np.dot(current_activation, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weight[i].T)

            if verbose:
                print("Derivative for W{}: {}".format(i, self.derivative[i]))

        return error


    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for j, input in enumerate(inputs):
                target = targets[j]

                # forward prop
                output = self.forward_propogate(input)

                # calculate error
                error = target - output

                # back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i+1))

    def gradient_descent(self, learning_rate=1):
        for i in range(len(self.weight)):
            weights = self.weight[i]
            derivatives = self.derivative[i]
            weights += derivatives * learning_rate

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target - output)**2)


if __name__ == "__main__":

    inputs = np.array([[random()/7 for _ in range(7)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create an MLP
    mlp = MLP(7, [7], 1)

    # train MLP
    mlp.train(inputs, targets, 100, 0.1)

    # creat dummy data
    input = np.array([[95], [95], [95], [95], [148], [149], [150], [150]])
    target = np.array([153])

    outputs = mlp.forward_propogate(input)
    print("Our prediction is: {}".format(outputs[0]))
