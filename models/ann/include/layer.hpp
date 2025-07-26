#pragma once

#include <vector>
#include <cstdint>
#include "neuron.hpp"

/**
 * @brief Represents a single layer in a neural network.
 */
class Layer {
public:
    /**
     * @brief Number of neurons in this layer.
     */
    int layer_size;

    /**
     * @brief Collection of pointers to neurons in the layer.
     */
    std::vector<Neuron *> neurons;

    /**
     * @brief Output values of the layer after activation.
     */
    std::vector<double> layer_outputs;

    /**
     * @brief Constructs a Layer with the given size.
     * @param prev_layer_size Number of neurons in the previous layer.
     * @param current_layer_size Number of neurons in the current layer.
     */
    Layer(int prev_layer_size, int current_layer_size);
};