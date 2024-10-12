#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_set>
#include "stdint.h"
#include "data.hpp"

class DataHandler {
    std::vector<Data *> *m_data_array; // All of the Data pre-split
    std::vector<Data *> *m_training_data;
    std::vector<Data *> *m_test_data;
    std::vector<Data *> *m_validation_data;

    int m_n_classes;
    int m_feature_vector_size;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

  public:
    DataHandler();
    ~DataHandler() {}

    void read_feature_vector(std::string path);
    void read_feature_labels(std::string path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char *bytes);

    std::vector<Data *> *get_training_data();
    std::vector<Data *> *get_test_data();
    std::vector<Data *> *get_validation_data();

};

#endif