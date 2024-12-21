#include <iostream>
#include "data_handler.hpp"

DataHandler::DataHandler() noexcept {
    m_data_array = new std::vector<Data *>;
    m_training_data = new std::vector<Data *>;
    m_test_data = new std::vector<Data *>;
    m_validation_data = new std::vector<Data *>;
}

DataHandler::~DataHandler() {
    delete m_data_array;
    delete m_training_data;
    delete m_test_data;
    delete m_validation_data;
}

void DataHandler::read_csv(std::string path, std::string delim) {
    m_n_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line; // Holds each line
    while (std::getline(data_file, line)) {
        if (line.length() == 0) {
            continue;
        }
        Data *d = new Data();
        d->set_double_feature_vector(new std::vector<double>());
        size_t pos = 0;
        std::string token; // Value in between delimiter
        while ((pos = line.find(delim)) != std::string::npos) {
            token = line.substr(0, pos);
            d->append_to_double_feature_vector(std::stod(token));
            line.erase(0, pos + delim.length());
        }
        if (str_class_map.find(line) != str_class_map.end()) {
            d->set_label(str_class_map[line]);
        } else {
            str_class_map[line] = m_n_classes;
            d->set_label(str_class_map[line]);
            m_n_classes++;
        }
        m_data_array->push_back(d);
        m_feature_vector_size = m_data_array->at(0)->get_double_feature_vector()->size();
    }
}

void DataHandler::read_feature_vector(const std::string& path) {
    uint32_t header[4]; // MAGIC | N_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (fp) {
        for (int i = 0; i < 4; ++i) {
            if (fread(bytes, sizeof(unsigned char), 4, fp) == 4) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Input File Header read completed." << std::endl;
        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; ++i) {
            Data *feature_vector = new Data();
            for (int j = 0; j < image_size; ++j) {
                uint8_t element[1];
                if (fread(element, sizeof(uint8_t), 1, fp) == 1) {
                    feature_vector->append_to_feature_vector(element[0]);
                } else {
                    std::cerr << "Error reading image data." << std::endl;
                    fclose(fp);
                    exit(1);
                }
            }
            m_data_array->push_back(feature_vector);
        }
        std::cout << "Successfully read and stored " << m_data_array->size() << " feature vectors." << std::endl;
        fclose(fp);
    } else {
        std::cerr << "Error opening file." << std::endl;
        exit(1);
    }
}

void DataHandler::read_feature_labels(const std::string& path) {
    uint32_t header[2]; // MAGIC | N_IMAGES
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (fp) {
        for (int i = 0; i < 2; ++i) {
            if (fread(bytes, sizeof(unsigned char), 4, fp) == 4) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Label File Header read completed." << std::endl;
        for (int i = 0; i < header[0]; ++i) {
            uint8_t element[1];
            if (fread(element, sizeof(uint8_t), 1, fp) == 1) {
                m_data_array->at(i)->set_label(element[0]);
            } else {
                std::cerr << "Error reading label data." << std::endl;
                fclose(fp);
                exit(1);
            }
        }
        std::cout << "Successfully read and stored labels." << std::endl;
        fclose(fp);
    } else {
        std::cerr << "Error opening file." << std::endl;
        exit(1);
    }
}

// Standalone function to select random data
void select_random_data(std::vector<Data *> *target_data, std::unordered_set<int> &used_indexes, int size, int count, std::vector<Data *> *data_array) {
    int current_count = 0;
    while (current_count < count) {
        int rand_index = rand() % size; // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            target_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            ++current_count;
        }
    }
}

void DataHandler::split_data() {
    std::unordered_set<int> used_indexes;

    auto size = m_data_array->size();
    int train_size = static_cast<int>(size * M_TRAIN_SET_PERCENT);
    int test_size = static_cast<int>(size * M_TEST_SET_PERCENT);
    int validation_size = static_cast<int>(size * M_VALIDATION_SET_PERCENT);

    select_random_data(m_training_data, used_indexes, size, train_size, m_data_array);
    select_random_data(m_test_data, used_indexes, size, test_size, m_data_array);
    select_random_data(m_validation_data, used_indexes, size, validation_size, m_data_array);


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

int DataHandler::get_class_counts() {
    return m_n_classes;
}

std::vector<Data *> *DataHandler::get_training_data() {
    return m_training_data;
}

std::vector<Data *> *DataHandler::get_validation_data() {
    return m_validation_data;
}

std::vector<Data *> *DataHandler::get_test_data() {
    return m_test_data;
}