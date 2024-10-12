#include <iostream>
#include <fstream>
#include <random>
#include "data_handler.hpp"

void DataHandler::read_feature_vector(const std::string& path) {
    uint32_t header[4]; // MAGIC | N_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening feature vector file.");
    }

    for (int i = 0; i < 4; ++i) {
        if (!file.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
            throw std::runtime_error("Error reading feature vector header.");
        }
        header[i] = convert_to_little_endian(bytes);
    }

    std::cout << "Input File Header read completed." << std::endl;
    int image_size = header[2] * header[3];
    for (int i = 0; i < header[1]; ++i) {
        Data feature_vector;
        uint8_t element;
        for (int j = 0; j < image_size; ++j) {
            if (!file.read(reinterpret_cast<char*>(&element), sizeof(element))) {
                throw std::runtime_error("Error reading image data.");
            }
            feature_vector.append_to_feature_vector(element);
        }
        m_data_array.push_back(std::move(feature_vector));
    }
    std::cout << "Successfully read and stored " << m_data_array.size() << " feature vectors." << std::endl;
}

void DataHandler::read_feature_labels(const std::string& path) {
    uint32_t header[2]; // MAGIC | N_IMAGES
    unsigned char bytes[4];
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening label file.");
    }

    for (int i = 0; i < 2; ++i) {
        if (!file.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
            throw std::runtime_error("Error reading label file header.");
        }
        header[i] = convert_to_little_endian(bytes);
    }

    std::cout << "Label File Header read completed." << std::endl;
    for (int i = 0; i < header[1]; ++i) {
        uint8_t element;
        if (!file.read(reinterpret_cast<char*>(&element), sizeof(element))) {
            throw std::runtime_error("Error reading label data.");
        }
        m_data_array.at(i).set_label(element);
    }
    std::cout << "Successfully read and stored labels." << std::endl;
}

// Standalone helper function to randomly select data from a source array
void select_random_data(const std::vector<Data>& source, std::vector<Data>& target, int target_size,
    std::unordered_set<int>& used_indexes, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, source.size() - 1);
    int count = 0;

    while (count < target_size) {
        int rand_index = dist(rng);
        if (used_indexes.insert(rand_index).second) {
            target.push_back(source.at(rand_index));
            ++count;
        }
    }
}

void DataHandler::split_data() {
    std::unordered_set<int> used_indexes;
    std::mt19937 rng(std::random_device{}());

    auto size = m_data_array.size();
    int train_size = static_cast<int>(size * M_TRAIN_SET_PERCENT);
    int test_size = static_cast<int>(size * M_TEST_SET_PERCENT);
    int validation_size = static_cast<int>(size * M_VALIDATION_SET_PERCENT);

    // Use the standalone helper function to select data
    select_random_data(m_data_array, m_training_data, train_size, used_indexes, rng);
    select_random_data(m_data_array, m_test_data, test_size, used_indexes, rng);
    select_random_data(m_data_array, m_validation_data, validation_size, used_indexes, rng);

    std::cout << "Training data size: " << m_training_data.size() << "." << std::endl;
    std::cout << "Test data size: " << m_test_data.size() << "." << std::endl;
    std::cout << "Validation data size: " << m_validation_data.size() << "." << std::endl;
}

void DataHandler::count_classes() {
    int count = 0;
    for (auto& data : m_data_array) {
        if (m_class_map.find(data.get_label()) == m_class_map.end()) {
            m_class_map[data.get_label()] = count;
            data.set_enumerated_label(count);
            ++count;
        }
    }
    m_n_classes = count;
    std::cout << "Successfully extracted " << m_n_classes << " unique classes." << std::endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char* bytes) const {
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

const std::vector<Data>& DataHandler::get_training_data() const {
    return m_training_data;
}

const std::vector<Data>& DataHandler::get_test_data() const {
    return m_test_data;
}

const std::vector<Data>& DataHandler::get_validation_data() const {
    return m_validation_data;
}