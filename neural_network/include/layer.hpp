#pragma once

#include <vector>
#include "neuron.hpp"

static int layer_id = 0;

class Layer {
public:
    int layer_size;
    std::vector<Neuron *> neurons;
    std::vector<double> layer_outputs;

    Layer(int prev_layer_size, int current_layer_size);
};