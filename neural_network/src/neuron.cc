#include <random>
#include "neuron.hpp"

double generate_random_number(double min, double max) {
    double random = (double) rand() / RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int prev_layer_size, int current_layer_size) {
    this->initialize_weights(prev_layer_size);
}

void Neuron::initialize_weights(int prev_layer_size) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < prev_layer_size + 1; ++i) {
        auto random_number = generate_random_number(-1.0, 1.0);
        this->weights.push_back(random_number);
    }
}