import numpy as np
import time

number_of_input_digits = 6

layers = [
    {
        "layerName": "input",
        "numberOfNodes": number_of_input_digits * 2,
        "activation": lambda x: x,
        "derivation": lambda x: x,
    },
    {
        "layerName": "hidden0",
        "numberOfNodes": 12,
        "activation": lambda x: np.tanh(x),
        "derivation": lambda x: 1 - np.tanh(x) ** 2,
    },
    {
        "layerName": "output",
        "numberOfNodes": number_of_input_digits,
        "activation": lambda x: sigmoid(x),
        "derivation": lambda x: x,
    },
]

number_of_training_samples = 300
number_of_training_epochs = 100
learning_rate = 0.01
momentum = 0.9
training_data = []


def init_training_data():
    input1_test_data = np.random.binomial(1, 0.5, (number_of_training_samples, number_of_input_digits))
    input2_test_data = np.random.binomial(1, 0.5, (number_of_training_samples, number_of_input_digits))

    for i in range(number_of_training_samples):
        current_target = np.logical_xor(input1_test_data[i], input2_test_data[i]).astype(int)
        current_data = {"input": [
            np.concatenate((input1_test_data[i], input2_test_data[i]), axis=0),
            input1_test_data[i],
            input2_test_data[i]
        ],
            "target": current_target
        }
        training_data.append(current_data)


def init_network():
    # hidden layer initialize
    for i in range(1, len(layers)):
        layers[i]["weights"] = np.random.normal(scale=0.1,
                                                size=(layers[i - 1]["numberOfNodes"], layers[i]["numberOfNodes"]))
        layers[i]["bias"] = [0]*layers[i]["numberOfNodes"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def predict(input_vector):
    result, result_activated = propagate_forward(input_vector)
    return np.array(result_activated[len(result_activated) - 1] > 0.5, dtype=int)


def propagate_forward(input_vector):
    temporary_vector = input_vector
    layer_result = []
    layer_result_activated = []

    for i in range(1, len(layers)):
        temporary_vector = np.dot(temporary_vector, layers[i]["weights"]) + layers[i]["bias"]
        layer_result.append(temporary_vector)
        layer_result_activated.append(layers[i]["activation"](temporary_vector))

    return layer_result, layer_result_activated


def propagate_backward(target, layer_results, layer_results_activated):
    weight_gradients = [] * (len(layers) - 1)
    bias_gradient = [0.0] * (len(layers) - 1)

    for l in range(1, len(layers)):
        weight_gradients.append([0.0] * layers[l]["numberOfNodes"])

    for l in reversed(range(1, len(layers))):
        for n in range(layers[l]["numberOfNodes"]):
            weight_gradients[l - 1][n] = calculate_gradient(l, n, weight_gradients, target, layer_results, layer_results_activated)

    # @todo calculate bias

    return weight_gradients, bias_gradient


def calculate_gradient(layer: int, node: int, weight_gradient, target, layer_results, layer_results_activated):

    # @todo

    return 0


def calculate_loss(target, result):
    loss = sum((target - result)**2 / 2)

    return loss


def train_epoch():
    epoch_errors = []
    epoch_weight_gradient = [] * (len(layers) - 1)

    for l in range(1, len(layers)):
        epoch_weight_gradient.append([0.0] * layers[l]["numberOfNodes"])

    for i in range(number_of_training_samples):

        current_input = training_data[i]["input"][0]
        current_target = training_data[i]["target"]

        layer_result, layer_result_activated = propagate_forward(current_input)
        current_result = layer_result_activated[-1]

        weight_gradients, bias_gradient = propagate_backward(current_target, layer_result, layer_result_activated)

        loss = calculate_loss(current_target, current_result)

        for j in range(len(layers) - 1):
            new_delta_gradient = np.array(weight_gradients[j], dtype=float) * learning_rate
            new_delta_gradient += momentum * bias_gradient[j]
            epoch_weight_gradient[j] -= new_delta_gradient

        epoch_errors.append(loss)

    # update layer weights
    for j in range(1, len(layers)):
        layers[j]["weights"] -= epoch_weight_gradient[j - 1]

    return epoch_errors


def train():
    for epoch in range(number_of_training_epochs):
        start_time = time.clock()

        epoch_errors = train_epoch()

        elapsed_time = time.clock() - start_time
        epoch_loss = np.asscalar(np.mean(epoch_errors))
        print("epoch: %d loss: %.8f time: %.4fs" % (epoch, epoch_loss, elapsed_time))
    print()


def test():
    # create random test data
    input1 = np.random.binomial(1, 0.5, number_of_input_digits)
    input2 = np.random.binomial(1, 0.5, number_of_input_digits)
    input_vector = np.concatenate((input1, input2), axis=0)
    output_vector = predict(input_vector)

    print("START TEST:")
    print("%s  (input 1)" % input1)
    print("%s  (input 2)" % input2)
    print("%s  (target)" % (np.logical_xor(input1, input2).astype(int)))
    print("%s  (output)" % output_vector)


init_training_data()

init_network()

train()

test()
