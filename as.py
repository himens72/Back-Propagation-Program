import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def layer_activation(data, weights, bias):
    return np.dot(data, weights) + bias


class NeuralNetwork:

    def __init__(self, i_data, o_data, input_count, hidden_count, output_count, lr):
        self.input = i_data
        self.output = o_data
        self.hidden_weights = np.random.uniform(size=(input_count, hidden_count))
        self.hidden_bias = np.random.uniform(size=(1, hidden_count))
        self.output_weights = np.random.uniform(size=(hidden_count, output_count))
        self.output_bias = np.random.uniform(size=(1, output_count))
        self.predicted_output = None
        self.learning_rate = lr
        self.error_iteration = []
        print("Question A : ")
        print("Initial hidden weights " + str(self.hidden_weights))
        print("Initial hidden bias " + str(self.hidden_bias))
        print("Initial output weights " + str(self.output_weights))
        print("Initial output bias " + str(self.output_bias))

    def predict(self, i_data):
        hidden_layer_output = sigmoid(layer_activation(i_data, self.hidden_weights, self.hidden_bias))
        output = sigmoid(layer_activation(hidden_layer_output, self.output_weights, self.output_bias))
        return output;

    def train(self):
        hidden_layer_output = sigmoid(layer_activation(self.input, self.hidden_weights, self.hidden_bias))
        self.predicted_output = sigmoid(layer_activation(hidden_layer_output, self.output_weights, self.output_bias))

        error = self.output - self.predicted_output
        delta_predicted_output = error * sigmoid_derivative(self.predicted_output)
        hidden_layer_error = delta_predicted_output.dot(self.output_weights.T)
        delta_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        self.error_iteration.append(error)
        # Updating Weights and Biases
        self.hidden_weights += self.input.T.dot(delta_hidden_layer) * learning_rate
        self.hidden_bias += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate
        self.output_weights += hidden_layer_output.T.dot(delta_predicted_output) * learning_rate
        self.output_bias += np.sum(delta_predicted_output, axis=0, keepdims=True) * learning_rate

    def final_data(self):
        print("Final hidden weights ", self.hidden_weights)
        print("Final hidden bias ", self.hidden_bias)
        print("Final output weights ", self.output_weights)
        print("Final output bias ", self.output_bias)
        print("\nPredicted Output from neural network after 10,000 epochs : ", self.predicted_output.T)

    def print_predicted_output(self):
        print("\nOutput from neural network for X = 0: \n", self.predicted_output)

    def get_hidden_weight(self):
        return self.hidden_weights

    def get_hidden_bias(self):
        return self.hidden_bias

    def get_output_weight(self):
        return self.output_weights

    def get_output_bias(self):
        return self.output_bias

    def get_error(self, index, sub_index):
        return self.error_iteration[index][sub_index]

    def get_output(self):
        return self.predicted_output


learning_rate = 0.1
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1
input_data = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_data, output_data, input_layer_neurons, hidden_layer_neurons, output_layer_neurons,
                   learning_rate)
epochs = 10000

iteration_count = []
class0_y1 = []
class0_y2 = []
class1_y1 = []
class1_y2 = []
for i in range(epochs):
    nn.train()
    iteration_count.append(i + 1)
    class0_y1.append(-1 * nn.get_error(i, 0))
    class0_y2.append(nn.get_error(i, 3))
    class1_y1.append(nn.get_error(i, 1))
    class1_y2.append(nn.get_error(i, 2))

nn.final_data()
class_label = []
threshold = 0.5
for i in range(len(nn.get_output())):
    if nn.get_output()[i] > 0.5:
        class_label.append([1])
    else:
        class_label.append([0])
print(
    "If predicted output is greater than threshold the it belongs to Class 1 otherwise it belongs to Class 0. Note Threshold is 0.5")
print(class_label)
print("\n-------------------------------------------------------------------------------\n ")
print("Question B : ")
print("Answer Shown in Figure 2")
print("\n-------------------------------------------------------------------------------\n ")
print("Question C : ")
print("Answer Shown in Figure 3")
predicted_output_for_X0 = nn.predict([[0, 0]])
if predicted_output_for_X0 > threshold:
    print('X = 0 belongs to Class 1 and predicted output is ', str(predicted_output_for_X0))
else:
    print('X = 0 belongs to Class 0 and predicted output is ', str(predicted_output_for_X0))

figure, ax = plt.subplots()
plt.plot(iteration_count, class0_y1)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epoch vs Error')
plt.figure()
X_axis = np.linspace(-1.5, 1.5)
boundary1 = (X_axis * nn.get_hidden_weight()[0, 0] + nn.get_hidden_bias()[0, 0]) / (-nn.get_hidden_weight()[0, 1])
boundary2 = (X_axis * nn.get_hidden_weight()[1, 0] + nn.get_hidden_bias()[0, 1]) / (-nn.get_hidden_weight()[1, 1])
plt.plot([-1, 1], [-1, 1], 'ro', label=' Class 0')
plt.plot([-1, 1], [1, -1], 'bo', label=' Class 1')
plt.plot(X_axis, boundary1)
plt.plot(X_axis, boundary2)
plt.legend()
plt.title('Question B : Final Decision Boundary')
plt.figure()
X_axis = np.linspace(-1.5, 1.5)
boundary1 = (X_axis * nn.get_hidden_weight()[0, 0] + nn.get_hidden_bias()[0, 0]) / (-nn.get_hidden_weight()[0, 1])
boundary2 = (X_axis * nn.get_hidden_weight()[1, 0] + nn.get_hidden_bias()[0, 1]) / (-nn.get_hidden_weight()[1, 1])
plt.plot([-1, 1], [-1, 1], 'ro', label=' Class 0')
plt.plot([-1, 1], [1, -1], 'bo', label=' Class 1')
plt.plot([0], [0], 'go', label='Point X = 0')
plt.plot(X_axis, boundary1)
plt.plot(X_axis, boundary2)
plt.legend()
plt.title('Question C : Decision Boundary at X = 0')
plt.show()
