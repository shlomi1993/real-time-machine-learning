#include <iostream>
#include <cstdio>
#include "data_handler.hpp"
#include "kmeans.hpp"

int main() {
    // Instantiate DataHandler and load the dataset
    DataHandler *dh = new DataHandler();
    dh->read_input_data("../../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../../dataset/train-labels-idx1-ubyte");
    dh->count_classes();
    dh->normalize();
    dh->split_data();

    // Initialize k
    double best_performance = 0.0;
    int best_k = 1;

    // Look for the best k
    for (int k = dh->get_class_count(); k < static_cast<int>(dh->get_training_set()->size() * 0.1); ++k) {

        // Instantiate KMeans with the current k
        auto *kmeans = new KMeans(k);
        kmeans->set_training_data(dh->get_training_set());
        kmeans->set_test_data(dh->get_test_set());
        kmeans->set_validation_data(dh->get_validation_set());

        // Initialize clusters and train the model
        kmeans->init_clusters();
        kmeans->train();

        // Validate the model
        double performance = kmeans->validate();
        std::cout << "Current Performance @ K = " << k << ": " << performance << std::endl;

        // Update k based on performance
        if (performance > best_performance) {
            best_performance = performance;
            best_k = k;
        }

        // Clean up
        delete kmeans;
    }

    // Retrain the a KMeans model with the best k
    std::cout << "Best K: " << best_k << " with performance: " << best_performance << std::endl;
    auto *final_kmeans = new KMeans(best_k);
    final_kmeans->set_training_data(dh->get_training_set());
    final_kmeans->set_test_data(dh->get_test_set());
    final_kmeans->set_validation_data(dh->get_validation_set());

    // Initialize clusters and train the final model
    final_kmeans->init_clusters();
    final_kmeans->train();
    std::cout << "Overall Performance: " << final_kmeans->test() << std::endl;

    // Clean up
    delete final_kmeans;
    delete dh;

    return 0;
}