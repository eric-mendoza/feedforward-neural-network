import numpy as np
import random


class Network(object):
    # Constructor
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # Initialize randomly
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # Initialize randomly

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data):
        training_data = list(training_data)
        n = len(training_data)
        test_data = list(test_data)
        n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            result = self.evaluate(test_data)
            print("Batch %s: %s/%s (%s%%)" % (j + 1, result, n_test, int(result/n_test * 100)))

    def update_mini_batch(self, mini_batch, eta):
        # 1) Training Sample
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 2) Feed, error, back-propagation
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 3) Gradient Descend
        # Update weights
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

        # Update biases
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, a, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 1) Input
        activation = a

        # 2) Feedforward
        activations = [a]  # Activations per layer
        zs = []  # Z-values per layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # Compute Z-value
            zs.append(z)  # Save Z

            activation = sigmoid(z)  # Update layer activation
            activations.append(activation)  # Save next layer activation (a)

        # 3) Cost Function
        output_layer = activations[-1]
        delta = cost_derivative(output_layer, y) * sigmoid_prime(zs[-1])  # Error in output
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 4) Back-propagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)  # Error based on next layer
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        # 5) Gradient for neural network cost
        return nabla_b, nabla_w

    def feedforward(self, a):
        # Feedforward input to classify
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        # Return last activation vector
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        correct_predictions = sum(int(x == y) for (x, y) in test_results)
        return correct_predictions

    # This class will evaluate a drawing
    def evaluate_drawing(self, drawing):
        result = self.feedforward(drawing)
        result_index = np.argmax(result)
        percentage = result[result_index] * 100
        return result_index, percentage


# This function saves the weights and biases
def save_network():
    pass


# THis function loads weights and biases
def load_network():
    pass


# This function returns the change on the quadratic cost function with respect to activation value
def cost_derivative(output_activations, y):
    return output_activations - y


# Sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Sigmoid derivative
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
