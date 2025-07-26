#pragma once

#include <vector>
#include "data_point.hpp"
#include "data_set.hpp"
#include "neuron.hpp"
#include "layer.hpp"

/**
 * @brief Represents a feedforward neural network with backpropagation training.
 */
class NeuralNetwork : public DataSet {
public:
    /**
     * @brief Layers of the neural network.
     */
    std::vector<Layer *> layers;

    /**
     * @brief Learning rate used during weight updates.
     */
    double learning_rate;

    /**
     * @brief Test set accuracy (percentage).
     */
    double test_performance;

    /**
     * @brief Constructs a neural network based on the given architecture.
     * @param spec Vector specifying number of neurons per layer.
     * @param input_size Number of input features.
     * @param output_size Number of output classes.
     * @param learning_rate Learning rate for training.
     */
    NeuralNetwork(std::vector<int> spec, int input_size, int output_size, double learning_rate);

    /**
     * @brief Destructor that cleans up allocated memory.
     */
    ~NeuralNetwork();

    /**
     * @brief Forward propagation through the network.
     * @param data Pointer to input data.
     * @return Output vector from the network.
     */
    std::vector<double> fprop(DataPoint *data_point);

    /**
     * @brief Computes dot product between inputs and weights.
     * @param inputs Input feature vector.
     * @param weights Weight vector.
     * @return Dot product result.
     */
    double activate(const std::vector<double> &inputs, const std::vector<double> &weights);

    /**
     * @brief Activation function (e.g., sigmoid).
     * @param x Input value.
     * @return Activated output.
     */
    double transfer(double x);

    /**
     * @brief Derivative of the activation function.
     * @param x Input value.
     * @return Derivative result.
     */
    double transfer_derivative(double x);

    /**
     * @brief Backward propagation of errors through the network.
     * @param data_point Pointer to the training data point.
     */
    void bprop(DataPoint *data_point);

    /**
     * @brief Updates weights based on the error terms.
     * @param data_point Pointer to the training data point.
     */
    void update_weights(DataPoint *data_point);

    /**
     * @brief Predicts the label for a given input.
     * @param data_point Pointer to the data point.
     * @return Predicted label (index of max output).
     */
    int predict(DataPoint *data_point);

    /**
     * @brief Trains the network for a given number of iterations.
     * @param iterations Number of training iterations.
     */
    void train(int iterations);

    /**
     * @brief Tests the model on the test set.
     * @return Accuracy as a percentage.
     */
    double test();

    /**
     * @brief Validates the model on the validation set.
     */
    void validate();
};