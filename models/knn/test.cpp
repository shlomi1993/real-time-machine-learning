#include <iostream>
#include "data_handler.hpp"
#include "knn.hpp"

int main() {
    // Initialize DataHandler and read input and label data
    DataHandler *dh = new DataHandler();
    dh->read_input_data("../../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../../dataset/train-labels-idx1-ubyte");
    dh->count_classes();
    dh->split_data();

    // Initialize KNN model
    KNN *knn = new KNN();
    knn->set_training_data(dh->get_training_data());
    knn->set_test_data(dh->get_test_data());
    knn->set_validation_data(dh->get_validation_data());

    // Find the best k value for KNN
    double performance = 0.0;
    double best_performance = 0.0;
    int best_k = 2;
    for (int k = best_k; k <= 10; k++) {
        knn->set_k(k);
        performance = knn->validate_performance();
        if (performance > best_performance) {
            best_performance = performance;
            best_k = k;
        }
    }

    // Test the KNN with the best k
    knn->set_k(best_k);
    knn->test_performance();

    // Clean up
    delete dh;
    delete knn;
}
