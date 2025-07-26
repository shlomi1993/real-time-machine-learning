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
#elif defined(IRIS) // Load IRIS dataset in CSV format
    dh->read_csv("../../dataset/iris.data", ",");
#else
#error "You must define either MNIST or IRIS to compile this code."
#endif

    // Preprocess: Normalize the data and split it into training, validation, and test sets
    dh->count_classes();
    dh->normalize();
    dh->split_data();

    // Get feature vector safely
    const auto& training_set = dh->get_training_set();
    if (training_set->empty()) {
        std::cerr << "[Fatal] Training data is empty!" << std::endl;
        delete dh;
        return 1;
    }

    // Ensure the first training data point has a normalized feature vector
    auto *vec_ptr = training_set->at(0)->get_normalized_feature_vector();
    if (!vec_ptr) {
        std::cerr << "[Fatal] Feature vector of first training data point is null!" << std::endl;
        delete dh;
        return 1;
    }

    // Define network architecture and parameters
    std::vector<int> hidden_layers = { 10, 10 };
    auto input_size = static_cast<int>(vec_ptr->size());
    auto output_size = dh->get_class_count();
    double learning_rate = 0.1;

    // Instantiate neural network
    NeuralNetwork* nn = new NeuralNetwork(hidden_layers, input_size, output_size, learning_rate);

    // Set datasets
    nn->set_training_data(training_set);
    nn->set_validation_data(dh->get_validation_set());
    nn->set_test_data(dh->get_test_set());

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