#include "../include/data_set.hpp"

void DataSet::set_training_data(std::vector<DataPoint *> *training_set) {
    this->training_set = training_set;
}

void DataSet::set_validation_data(std::vector<DataPoint *> *validation_set) {
    this->validation_set = validation_set;
}

void DataSet::set_test_data(std::vector<DataPoint *> *test_set) {
    this->test_set = test_set;
}
