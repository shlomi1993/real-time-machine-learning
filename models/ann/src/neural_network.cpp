#include <numeric>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "data_handler.hpp"
#include "layer.hpp"
#include "neural_network.hpp"

/**
 * @brief Constructs the network based on layer specification.
 * @param spec Vector containing number of neurons in each hidden layer.
 * @param input_size Number of input features.
 * @param num_classes Number of output classes.
 * @param learning_rate Learning rate for weight updates.
 */
NeuralNetwork::NeuralNetwork(std::vector<int> spec, int input_size, int num_classes, double learning_rate)
    : learning_rate(learning_rate), test_performance(0.0)
{
    for (size_t i = 0; i < spec.size(); ++i) {
        if (i == 0) {
            // First hidden layer connected to input layer
            layers.push_back(new Layer(input_size, spec.at(i)));
        } else {
            // Subsequent hidden layers connected to previous hidden layer
            layers.push_back(new Layer(static_cast<int>(layers.at(i - 1)->neurons.size()), spec.at(i)));
        }
    }
    // Output layer connected to last hidden layer
    layers.push_back(new Layer(static_cast<int>(layers.back()->neurons.size()), num_classes));
}

/**
 * @brief Destructor that releases dynamically allocated neurons and layers.
 */
NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        for (Neuron* neuron : layer->neurons) {
            delete neuron;
        }
        delete layer;
    }
    layers.clear();
}

/**
 * @brief Computes the activation for a neuron as weighted sum + bias.
 * @param weights Weights vector including bias as last element.
 * @param inputs Input feature vector.
 * @return Activation value.
 */
double NeuralNetwork::activate(const std::vector<double> &weights, const std::vector<double> &inputs) {
    double activation = weights.back(); // Bias term
    for (size_t i = 0; i < weights.size() - 1; ++i) {
        activation += weights[i] * inputs[i];
    }
    return activation;
}

/**
 * @brief Sigmoid activation function.
 * @param activation Raw activation input.
 * @return Output after sigmoid.
 */
double NeuralNetwork::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

/**
 * @brief Derivative of sigmoid function used in backpropagation.
 * @param output Output from sigmoid function.
 * @return Derivative value.
 */
double NeuralNetwork::transfer_derivative(double output) {
    return output * (1.0 - output);
}

/**
 * @brief Forward propagation through all layers.
 * @param data Input data point.
 * @return Output vector from final layer.
 */
std::vector<double> NeuralNetwork::fprop(DataPoint *data) {
    // Start with input features
    std::vector<double> inputs = *data->get_normalized_feature_vector();

    // Propagate through each layer
    for (Layer* layer : layers) {
        std::vector<double> new_inputs;

        // Compute output of each neuron in the layer
        for (Neuron* neuron : layer->neurons) {
            double activation = activate(neuron->weights, inputs);
            neuron->output = transfer(activation);
            new_inputs.push_back(neuron->output);
        }

        inputs = std::move(new_inputs);  // outputs become inputs for next layer
    }

    return inputs;  // final output layer's outputs
}

/**
 * @brief Backward propagation of error and calculation of delta values.
 * @param data Training data point.
 */
void NeuralNetwork::bprop(DataPoint *data_point) {
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        Layer* layer = layers.at(i);
        std::vector<double> errors;

        if (i != static_cast<int>(layers.size()) - 1) {
            // Hidden layers: error is weighted sum of next layer deltas
            for (size_t j = 0; j < layer->neurons.size(); ++j) {
                double error = 0.0;
                Layer* next_layer = layers.at(i + 1);
                for (Neuron* neuron : next_layer->neurons) {
                    error += neuron->weights[j] * neuron->delta;
                }
                errors.push_back(error);
            }
        } else {
            // Output layer: error is difference between expected and actual output
            for (size_t j = 0; j < layer->neurons.size(); ++j) {
                Neuron* neuron = layer->neurons.at(j);
                errors.push_back(static_cast<double>(data_point->get_class_vector().at(j)) - neuron->output);
            }
        }

        // Calculate delta = error * derivative of activation function
        for (size_t j = 0; j < layer->neurons.size(); ++j) {
            Neuron* neuron = layer->neurons.at(j);
            neuron->delta = errors[j] * transfer_derivative(neuron->output);
        }
    }
}

/**
 * @brief Updates weights based on delta values and learning rate.
 * @param data Training data point.
 */
void NeuralNetwork::update_weights(DataPoint *data_point) {
    // Start with input features for first layer
    std::vector<double> inputs = *data_point->get_normalized_feature_vector();

    for (size_t i = 0; i < layers.size(); ++i) {
        if (i != 0) {
            // For layers beyond first, inputs are outputs from previous layer
            Layer* prev_layer = layers.at(i - 1);
            for (Neuron* neuron : prev_layer->neurons) {
                inputs.push_back(neuron->output);
            }
        }

        Layer* layer = layers.at(i);
        for (Neuron* neuron : layer->neurons) {
            // Update weights including bias (last weight)
            for (size_t j = 0; j < inputs.size(); ++j) {
                neuron->weights[j] += learning_rate * neuron->delta * inputs[j];
            }
            neuron->weights.back() += learning_rate * neuron->delta;  // bias update
        }

        inputs.clear();
    }
}

/**
 * @brief Predicts label for input data by selecting neuron with max output.
 * @param data Input data point.
 * @return Index of predicted class.
 */
int NeuralNetwork::predict(DataPoint *data_point) {
    std::vector<double> outputs = fprop(data_point);
    return static_cast<int>(std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())));
}

/**
 * @brief Trains the network for a specified number of epochs.
 * @param num_epochs Number of training iterations over the entire training set.
 */
void NeuralNetwork::train(int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double sum_error = 0.0;

        for (DataPoint* data_point : *training_set) {
            std::vector<double> outputs = fprop(data_point);
            std::vector<int> expected = data_point->get_class_vector();

            // Compute sum squared error for current data point
            double error_sum = 0.0;
            for (size_t j = 0; j < outputs.size(); ++j) {
                double diff = static_cast<double>(expected[j]) - outputs[j];
                error_sum += diff * diff;
            }
            sum_error += error_sum;

            bprop(data_point);
            update_weights(data_point);
        }

        std::printf("Epoch: %d \t Error=%.4f\n", epoch, sum_error);
    }
}

/**
 * @brief Tests the network on the test dataset.
 * @return Accuracy (fraction of correctly predicted samples).
 */
double NeuralNetwork::test() {
    double num_correct = 0.0;
    double count = 0.0;

    for (DataPoint* data_point : *test_set) {
        ++count;
        int prediction = predict(data_point);
        if (data_point->get_class_vector().at(prediction) == 1) {
            ++num_correct;
        }
    }

    test_performance = num_correct / count;
    return test_performance;
}

/**
 * @brief Validates the network on the validation dataset and prints performance.
 */
void NeuralNetwork::validate() {
    double num_correct = 0.0;
    double count = 0.0;

    for (DataPoint* data_point : *validation_set) {
        ++count;
        int prediction = predict(data_point);
        if (data_point->get_class_vector().at(prediction) == 1) {
            ++num_correct;
        }
    }

    std::printf("Validation Performance: %.4f\n", num_correct / count);
}