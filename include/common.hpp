#pragma once

#include <iostream>
#include <vector>
#include "data.hpp"


class CommonData {
protected:
    std::vector<Data *> *m_training_data;
    std::vector<Data *> *m_validation_data;
    std::vector<Data *> *m_test_data;

public:
    void set_training_data(std::vector<Data *> *training_data);
    void set_validation_data(std::vector<Data *> *validation_data);
    void set_test_data(std::vector<Data *> *test_data);
};