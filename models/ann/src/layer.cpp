#include "layer.hpp"

/**
 * @brief Constructs a Layer with neurons initialized based on previous layer size.
 * @param prev_layer_size Number of neurons in the previous layer.
 * @param current_layer_size Number of neurons in the current layer.
 */
Layer::Layer(int prev_layer_size, int current_layer_size)
    : layer_size(current_layer_size)
{
    neurons.reserve(current_layer_size);
    for (int i = 0; i < current_layer_size; ++i) {
        neurons.push_back(new Neuron(prev_layer_size));
    }
}