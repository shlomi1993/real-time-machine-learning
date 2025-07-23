#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <cstdint>
#include <memory>
#include <fstream>
#include "data_point.hpp"

/**
 * @brief Handles reading, preprocessing, and splitting of datasets.
 */
class DataHandler {
private:
    const double TRAIN_SET_PERCENT = 0.75;
    const double VALIDATION_SET_PERCENT = 0.05;
    const double TEST_SET_PERCENT = 0.20;

    std::vector<DataPoint *> *data_array = nullptr;
    std::vector<DataPoint *> *training_data = nullptr;
    std::vector<DataPoint *> *validation_data = nullptr;
    std::vector<DataPoint *> *test_data = nullptr;

    int num_classes = 0;
    uint8_t feature_vector_size = 0;
    std::map<uint8_t, int> class_map;
    std::map<std::string, int> str_class_map;

public:
    /**
     * @brief Constructs a new DataHandler object.
     */
    DataHandler() noexcept;

    /**
     * @brief Destroys the DataHandler object and frees all data pointers.
     */
    ~DataHandler();

    /**
     * @brief Reads data from a CSV file with a custom delimiter.
     * @param path Path to the CSV file.
     * @param delim Delimiter used in the CSV file.
     */
    void read_csv(const std::string &path, const std::string &delim);

    /**
     * @brief Reads data from a CSV file using the default delimiter (comma).
     * @param path Path to the CSV file.
     */
    void read_csv(const std::string &path);

    /**
     * @brief Reads binary input data (e.g., feature vectors like MNIST images).
     * @param path Path to the binary file.
     */
    void read_input_data(const std::string &path);

    /**
     * @brief Reads binary label data (e.g., class labels like MNIST labels).
     * @param path Path to the binary label file.
     */
    void read_label_data(const std::string &path);

    /**
     * @brief Splits the dataset into training, validation, and test sets.
     */
    void split_data();

    /**
     * @brief Counts the number of unique class labels in the dataset.
     */
    void count_classes();

    /**
     * @brief Converts a 4-byte array into a 32-bit little-endian integer.
     * @param bytes Pointer to a 4-byte array.
     * @return Converted 32-bit unsigned integer.
     */
    uint32_t convert_to_little_endian(const unsigned char *bytes);

    /**
     * @brief Normalizes all feature vectors to the range [0, 1].
     */
    void normalize();

    /**
     * @brief Returns the total number of unique classes in the dataset.
     * @return Number of classes.
     */
    int get_class_count() const;

    /**
     * @brief Returns a pointer to the training data set.
     * @return Pointer to vector of DataPoint*.
     */
    std::vector<DataPoint *> *get_training_data() const;

    /**
     * @brief Returns a pointer to the validation data set.
     * @return Pointer to vector of DataPoint*.
     */
    std::vector<DataPoint *> *get_validation_data() const;

    /**
     * @brief Returns a pointer to the test data set.
     * @return Pointer to vector of DataPoint*.
     */
    std::vector<DataPoint *> *get_test_data() const;
};