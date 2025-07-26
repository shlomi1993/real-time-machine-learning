#include <iostream>
#include "data_handler.hpp"

void assert_equal(int a, int b, const std::string &msg) {
    if (a != b) {
        throw std::runtime_error("Assertion failed: " + msg);
    }
}

// Unittest - ETL
int main() {
    DataHandler *dh = new DataHandler();
    dh->read_input_data("../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    const size_t total = 60000;
    const size_t expected_train_size = static_cast<size_t>(total * 0.75);
    const size_t expected_val_size = static_cast<size_t>(total * 0.05);
    const size_t expected_test_size = total - expected_train_size - expected_val_size;

    std::cout << std::endl;
    assert_equal(dh->get_class_count(), 10, "Class count should be 10");
    assert_equal(dh->get_training_set()->size(), expected_train_size, "Training data size mismatch");
    assert_equal(dh->get_validation_set()->size(), expected_val_size, "Validation data size mismatch");
    assert_equal(dh->get_test_set()->size(), expected_test_size, "Test data size mismatch");
    std::cout << "All tests passed successfully!" << std::endl;

    return 0;
}