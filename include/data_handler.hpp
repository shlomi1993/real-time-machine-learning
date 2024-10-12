#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include "data.hpp"

class DataHandler {
    std::vector<Data> m_data_array; // All of the Data pre-split
    std::vector<Data> m_training_data;
    std::vector<Data> m_test_data;
    std::vector<Data> m_validation_data;

    int m_n_classes = 0;
    int m_feature_vector_size = 0;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

public:
    DataHandler() = default;
    void read_feature_vector(const std::string& path);
    void read_feature_labels(const std::string& path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char* bytes) const;

    const std::vector<Data>& get_training_data() const;
    const std::vector<Data>& get_test_data() const;
    const std::vector<Data>& get_validation_data() const;
};

#endif // DATA_HANDLER_HPP