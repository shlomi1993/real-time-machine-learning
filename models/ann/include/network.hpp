#pragma once

#include "data_set.hpp"
#include "data_point.hpp"
#include "neuron.hpp"
#include "layer.hpp"

class Network : public DataSet {
    std::vector<Layer *> layers;
    double learning_rate;
    double test_performance;

public:
    Network(std::vector<int> spec, int input_size, int n_classes, double learning_rate);
    double activate(std::vector<double> weights, std::vector<double> inputs);  // Dot product
    double inline transfer(double);
    double inline transfer_derivative(double);  // Used for Back-Propagation
    std::vector<double> fprop(DataPoint *data);
    void bprop(DataPoint *data);
    void update_weights(DataPoint *data);
    int predict(DataPoint *data);
    void train(int n_iterations);
    double validate();
    double test();
};