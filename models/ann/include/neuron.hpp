#pragma once

#include <vector>
#include <cmath>
#include <cstdio>

/**
 * @brief Represents a single neuron in a neural network.
 */
class Neuron {
public:
    /**
     * @brief Output value after activation.
     */
    double output;

    /**
     * @brief Error term used in backpropagation.
     */
    double delta;

    /**
     * @brief Weights associated with inputs to this neuron.
     */
    std::vector<double> weights;

    /**
     * @brief Constructs a Neuron and initializes its weights.
     * @param prev_layer_size Number of inputs to this neuron (previous layer size).
     */
    Neuron(int prev_layer_size);

    /**
     * @brief Initializes weights with random values.
     * @param num_inputs Number of inputs to initialize weights for.
     */
    void initialize_weights(int num_inputs);
};