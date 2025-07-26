#include <random>
#include "neuron.hpp"

/**
 * @brief Constructs a Neuron and initializes its weights.
 * @param prev_layer_size Number of inputs to this neuron (previous layer size).
 */
Neuron::Neuron(int prev_layer_size)
{
    initialize_weights(prev_layer_size);
}

/**
 * @brief Initializes weights with random values uniformly distributed in [-1.0, 1.0].
 * Adds one extra weight for the bias term.
 * @param prev_layer_size Number of inputs to initialize weights for.
 */
void Neuron::initialize_weights(int prev_layer_size)
{
    static std::random_device rd;  // only seeded once
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    weights.clear();
    for (int i = 0; i < prev_layer_size + 1; ++i) {
        weights.push_back(dist(gen));
    }
}