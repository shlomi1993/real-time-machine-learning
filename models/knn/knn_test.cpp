#include <iostream>
#include "data_handler.hpp"
#include "knn.hpp"

int main() {
    DataHandler *dh = new DataHandler();
    dh->read_input_data("../../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../../dataset/train-labels-idx1-ubyte");
    dh->count_classes();
    dh->split_data();
    KNN *nearest = new KNN();
    nearest->set_k(3);
    nearest->set_training_data(dh->get_training_data());
    nearest->set_test_data(dh->get_test_data());
    nearest->set_validation_data(dh->get_validation_data());
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for (int k = 1; k <= 3; k++) {
        if (k == 1) {
        performance = nearest->validate_performance();
        best_performance = performance;
        } else {
            nearest->set_k(k);
            performance = nearest->validate_performance();
            if (performance > best_performance) {
                best_performance = performance;
                best_k = k;
            }
        }
    }
    nearest->set_k(best_k);
    nearest->test_performance();

    delete dh;
    delete nearest;
}
