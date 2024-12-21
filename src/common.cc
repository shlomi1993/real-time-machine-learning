#include "common.hpp"

void CommonData::set_training_data(std::vector<Data *> *training_data) {
    m_training_data = training_data;
}

void CommonData::set_validation_data(std::vector<Data *> *validation_data) {
    m_validation_data = validation_data;
}

void CommonData::set_test_data(std::vector<Data *> *test_data) {
    m_test_data = test_data;
}