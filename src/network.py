# Feedforward neural network
# Network based in book available in: http://neuralnetworksanddeeplearning.com/chap2.html

import numpy as np
import random as rnd


class Network(object):
    # Constructor
    def __init__(self, layers):
        # Basic parameters of Neural Network
        self.layers = layers
        self.biases = []

        # Create an array for the biases. Each index is a column vector with initial random numbers
        for no_neurons in layers[1:]:
            self.biases.append(np.random.randn(no_neurons, 1))

        # Create an array for the weights. Each index is a
        self.weights = []
        for layer in range(len(layers) - 1):
            self.weights.append(np.random.rand(layers[layer+1], layers[layer])/np.sqrt(layers[layer]))

        print("Nice, Neural Network Created!")

    # The training is going to be done with Stochastic Gradient Descend, which means we will use batches to do execute
    # normal gradient descend.
    def train(self, train, epochs, batch_size, learning_rate, test, test_train=False):
        # Parse Zip object to list of tuples
        train = list(train)
        test = list(test)

        # Use the train data several times making sure to not overfit or underfit
        print("Training is about to start.")
        for epoch in range(epochs):
            # Create batches for gradient descend
            rnd.shuffle(train)  # Shuffle training data again
            batches = []
            for training_tuple in range(0, len(train), batch_size):
                batches.append(train[training_tuple:training_tuple + batch_size])

            # This updates the weights and biases with the average of the derivatives obtained in the batches
            for batch in batches:
                self.gradient_descend(batch, learning_rate)

            result = self.evaluate(test)
            print("%s) Test: %s/%s (%s%%)" % (epoch + 1, result, len(test), int(result/len(test) * 100)))

    def gradient_descend(self, batch, learning_rate):
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))

        # Feed, error, back-propagation
        for image, result in batch:
            # Backpropagation returns the delta
            delta_nabla_b, delta_nabla_w = self.backpropagation(image, result)

            # Calculate nabla_b sum
            for i in range(len(delta_nabla_b)):
                nabla_b[i] += delta_nabla_b[i]

            # Calculate weight nabla sum
            for i in range(len(delta_nabla_w)):
                nabla_w[i] += delta_nabla_w[i]

        # Update weights substracting the mean of nabla vectors. That's because we want to go down the hill
        for i in range(len(nabla_w)):
            self.weights[i] = self.weights[i] - (learning_rate * nabla_w[i]/len(batch))

        # Update biases
        for i in range(len(nabla_b)):
            self.biases[i] = self.biases[i] - (learning_rate * nabla_b[i]/len(batch))

    def backpropagation(self, image, result):
        delta_nabla_b = []
        for b in self.biases:
            delta_nabla_b.append(np.zeros(b.shape))
        delta_nabla_w = []
        for w in self.weights:
            delta_nabla_w.append(np.zeros(w.shape))

        # 1) Input
        activation = image

        # 2) Feedforward
        activations = [image]  # Activations per layer
        zs = []  # Z-values per layer
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], activation) + self.biases[i]  # Compute Z-value
            zs.append(z)  # Save Z
            activation = sigmoid(z)  # Update layer activation for next layer
            activations.append(activation)  # Save next layer activation (a)

        # 3) Cost Function Derivative
        delta = cost_derivative(activations[-1], result) * sigmoid_prime(zs[-1])  # Eq 1: d = nablaC (.) sigma_prime
        delta_nabla_b[-1] = delta  # Equation 3: dC/dB = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # Eq 4: delta * a

        # 4) Back-propagate the error
        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])  # Error based on next layer
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        # 5) Return Gradient for neural network cost
        return delta_nabla_b, delta_nabla_w

    def feedforward(self, image):
        # Feedforward input to classify
        for i in range(len(self.biases)):
            image = sigmoid(np.dot(self.weights[i], image) + self.biases[i])

        # Return last activation vector
        return image

    def evaluate(self, test_data):
        correct_predictions = 0
        for image, expected in test_data:
            correct_predictions += int(np.argmax(self.feedforward(image)) == np.argmax(expected))
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
