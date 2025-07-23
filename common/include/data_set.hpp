#pragma once

#include <vector>
#include "data_point.hpp"

/**
 * @brief Abstract base class for models using training, validation, and test data.
 */
class DataSet {
protected:
    std::vector<DataPoint*>* training_set = nullptr;
    std::vector<DataPoint*>* validation_set = nullptr;
    std::vector<DataPoint*>* test_set = nullptr;

public:
    virtual ~DataSet() = default;

    /**
     * @brief Set the training dataset.
     * @param training_set Pointer to vector of DataPoint pointers.
     */
    void set_training_data(std::vector<DataPoint*> *training_set);

    /**
     * @brief Set the validation dataset.
     * @param validation_set Pointer to vector of DataPoint pointers.
     */
    void set_validation_data(std::vector<DataPoint*> *validation_set);

    /**
     * @brief Set the test dataset.
     * @param test_set Pointer to vector of DataPoint pointers.
     */
    void set_test_data(std::vector<DataPoint*> *test_set);
};