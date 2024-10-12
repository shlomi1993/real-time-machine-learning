#include <iostream>
#include "data_handler.hpp"

DataHandler::DataHandler() {
    m_data_array = new std::vector<Data *>;
    m_training_data = new std::vector<Data *>;
    m_test_data = new std::vector<Data *>;
    m_validation_data = new std::vector<Data *>;
}

void DataHandler::read_feature_vector(std::string path) {
    uint32_t header[4]; // MAGIC | N_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (fp) {
        for (int i = 0; i < 4; ++i) {
            if (fread(bytes, sizeof(bytes), 1, fp)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Input File Header read completed." << std::endl;
        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; ++i) {
            Data *feature_vector = new Data();
            uint8_t element[1];
            for (int j = 0; j < image_size; ++j) {
                if (fread(element, sizeof(element), 1, fp)) {
                    feature_vector->append_to_feature_vector(element[0]);
                } else {
                    std::cerr << "Error reading image data." << std::endl;
                    exit(1);
                }
            }
            m_data_array->push_back(feature_vector);
        }
        std::cout << "Successfully read and stored " << m_data_array->size() << " feature vectors." << std::endl;
    } else {
        std::cerr << "Error opening file." << std::endl;
        exit(1);
    }
}

void DataHandler::read_feature_labels(std::string path) {
    uint32_t header[2]; // MAGIC | N_IMAGES
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (fp) {
        for (int i = 0; i < 2; ++i) {
            if (fread(bytes, sizeof(bytes), 1, fp)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Label File Header read completed." << std::endl;
        for (int i = 0; i < header[0]; ++i) {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, fp)) {
                m_data_array->at(i)->set_label(element[0]);
            } else {
                std::cerr << "Error reading image data." << std::endl;
                exit(1);
            }
        }
        std::cout << "Successfully read and stored labels." << std::endl;
    } else {
        std::cerr << "Error opening file." << std::endl;
        exit(1);
    }
}

void DataHandler::split_data() {
    std::unordered_set<int> used_indexes;

    auto size = m_data_array->size();
    int train_size = size * TRAIN_SET_PERCENT;
    int test_size = size * TEST_SET_PERCENT;
    int validation_size = size * VALIDATION_SET_PERCENT;

    // Training data
    int count = 0;
    while (count < train_size) {
        int rand_index = rand() % size; // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            m_training_data->push_back(m_data_array->at(rand_index));
            used_indexes.insert(rand_index);
            ++count;
        }
    }

    // Test data
    count = 0;
    while (count < test_size) {
        int rand_index = rand() % size; // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            m_test_data->push_back(m_data_array->at(rand_index));
            used_indexes.insert(rand_index);
            ++count;
        }
    }

    // Validation data
    count = 0;
    while (count < validation_size) {
        int rand_index = rand() % size; // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            m_validation_data->push_back(m_data_array->at(rand_index));
            used_indexes.insert(rand_index);
            ++count;
        }
    }

    std::cout << "Training data size: " << m_training_data->size() << "." << std::endl;
    std::cout << "Test data size: " << m_test_data->size() << "." << std::endl;
    std::cout << "Validation data size: " << m_validation_data->size() << "." << std::endl;
}

void DataHandler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < m_data_array->size(); ++i) {
        if (class_map.find(m_data_array->at(i)->get_label()) == class_map.end()) {
            class_map[m_data_array->at(i)->get_label()] = count;
            m_data_array->at(i)->set_enumerated_label(count);
            ++count;
        }
    }
    m_n_classes = count;
    std::cout << "Successfully extracted " << m_n_classes << " unique classes." << std::endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char *bytes) {
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3] << 0));
}

std::vector<Data *> *DataHandler::get_training_data() {
    return m_training_data;
}

std::vector<Data *> *DataHandler::get_test_data() {
    return m_test_data;
}

std::vector<Data *> *DataHandler::get_validation_data() {
    return m_validation_data;
}