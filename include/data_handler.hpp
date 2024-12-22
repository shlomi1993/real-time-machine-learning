#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include "data.hpp"

class DataHandler {
    std::vector<Data *> *data_array; // All of the Data pre-split
    std::vector<Data *> *training_data;
    std::vector<Data *> *validation_data;
    std::vector<Data *> *test_data;

    int n_classes = 0;
    uint8_t feature_vector_size = 0;
    std::map<uint8_t, int> class_map;
    std::map<std::string, int> str_class_map; // For DNN models

    const double TRAIN_SET_PERCENT = 0.75;
    const double VALIDATION_SET_PERCENT = 0.05;
    const double TEST_SET_PERCENT = 0.20;

  public:
    DataHandler() noexcept;
    ~DataHandler();

    void read_csv(std::string path, std::string delim);
    void read_csv(std::string path);
    void read_feature_vector(const std::string& path);
    void read_feature_labels(const std::string& path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char *bytes);
    void normalize();

    int get_class_counts();
    std::vector<Data *> *get_training_data();
    std::vector<Data *> *get_validation_data();
    std::vector<Data *> *get_test_data();
};