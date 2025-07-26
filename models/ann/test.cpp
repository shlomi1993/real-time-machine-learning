#include <iostream>
#include <vector>
#include "data_handler.hpp"
#include "neural_network.hpp"

int main() {
    // Create and initialize data handler
    DataHandler* dh = new DataHandler();

#if defined(MNIST) // Load MNIST binary format
    dh->read_input_data("../../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../../dataset/train-labels-idx1-ubyte");
    dh->count_classes();
#elif defined(IRIS) // Load IRIS dataset in CSV format
    dh->read_csv("../../dataset/iris.data", ",");
#else
#error "You must define either MNIST or IRIS to compile this code."
#endif

    // Split into training, validation, and test sets
    dh->split_data();

    // Define network architecture: one hidden layer with 10 neurons
    std::vector<int> hidden_layers = {10};

    // Instantiate neural network
    NeuralNetwork* nn = new NeuralNetwork(
        hidden_layers,
        static_cast<int>(dh->get_training_data()->at(0)->get_normalized_feature_vector()->size()), // input size
        dh->get_class_count(),  // number of output classes
        0.25 // learning rate
    );

    // Set datasets
    nn->set_training_data(dh->get_training_data());
    nn->set_validation_data(dh->get_validation_data());
    nn->set_test_data(dh->get_test_data());

    // Train the model
    nn->train(15);

    // Validate the model
    nn->validate();

    // Test the model
    std::cout << "Test Performance: " << nn->test() << std::endl;

    // Cleanup
    delete nn;
    delete dh;

    return 0;
}