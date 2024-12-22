#pragma once

#include <map>

class Neuron {
public:
    std::vector<double> weights;
    double delta;
    double output;

    Neuron(int, int);
    void initialize_weights(int prev_layer_size);
};