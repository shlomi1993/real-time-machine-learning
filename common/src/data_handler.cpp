#include <iostream>
#include <cstdlib>
#include <unordered_set>
#include "data_handler.hpp"

// Constructor
DataHandler::DataHandler() noexcept {
    data_array = new std::vector<DataPoint *>;
    training_data = new std::vector<DataPoint *>;
    test_data = new std::vector<DataPoint *>;
    validation_data = new std::vector<DataPoint *>;
}

// Destructor
DataHandler::~DataHandler() {
    delete data_array;
    delete training_data;
    delete test_data;
    delete validation_data;
}

void DataHandler::read_csv(const std::string &path, const std::string &delim) {
    this->num_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line;

    while (std::getline(data_file, line)) {
        if (line.empty()) continue;

        auto *d = new DataPoint();
        d->set_normalized_feature_vector(new std::vector<double>());

        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delim)) != std::string::npos) {
            token = line.substr(0, pos);
            d->append_to_normalized_feature_vector(std::stod(token));
            line.erase(0, pos + delim.length());
        }

        if (str_class_map.find(line) != str_class_map.end()) {
            d->set_label(str_class_map[line]);
        } else {
            str_class_map[line] = num_classes;
            d->set_label(num_classes++);
        }

        data_array->push_back(d);
        feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
    }
}

void DataHandler::read_csv(const std::string &path) {
    read_csv(path, ",");
}

void DataHandler::read_input_data(const std::string &path) {
    uint32_t header[4];
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp) {
        std::cerr << "Error opening input file '" << path << "': ";
        perror(nullptr);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 4; ++i) {
        if (fread(bytes, sizeof(unsigned char), 4, fp) == 4) {
            header[i] = convert_to_little_endian(bytes);
        }
    }

    std::cout << "Input File Header read completed." << std::endl;

    int image_size = header[2] * header[3];
    for (uint32_t i = 0; i < header[1]; ++i) {
        auto *dp = new DataPoint();
        for (int j = 0; j < image_size; ++j) {
            uint8_t element[1];
            if (fread(element, sizeof(uint8_t), 1, fp) != 1) {
                std::cerr << "Error reading image data." << std::endl;
                fclose(fp);
                exit(1);
            }
            dp->append_to_feature_vector(element[0]);
        }
        data_array->push_back(dp);
    }

    std::cout << "Successfully read and stored " << data_array->size() << " feature vectors." << std::endl;
    fclose(fp);
}

void DataHandler::read_label_data(const std::string &path) {
    uint32_t header[2];
    unsigned char bytes[4];
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp) {
        std::cerr << "Error opening label file." << std::endl;
        exit(1);
    }

    for (int i = 0; i < 2; ++i) {
        if (fread(bytes, sizeof(unsigned char), 4, fp) == 4) {
            header[i] = convert_to_little_endian(bytes);
        }
    }

    std::cout << "Label File Header read completed." << std::endl;

    for (uint32_t i = 0; i < header[1]; ++i) {
        uint8_t element[1];
        if (fread(element, sizeof(uint8_t), 1, fp) != 1) {
            std::cerr << "Error reading label data." << std::endl;
            fclose(fp);
            exit(1);
        }
        data_array->at(i)->set_label(element[0]);
    }

    std::cout << "Successfully read and stored labels." << std::endl;
    fclose(fp);
}

static void select_random_data(std::vector<DataPoint *> *target, std::unordered_set<int> &used, int total, int count, std::vector<DataPoint *> *source) {
    int current = 0;
    while (current < count) {
        int idx = rand() % total;
        if (used.find(idx) == used.end()) {
            target->push_back(source->at(idx));
            used.insert(idx);
            ++current;
        }
    }
}

void DataHandler::split_data() {
    std::unordered_set<int> used;

    int total = static_cast<int>(data_array->size());
    int train_count = static_cast<int>(total * TRAIN_SET_PERCENT);
    int test_count = static_cast<int>(total * TEST_SET_PERCENT);
    int val_count = static_cast<int>(total * VALIDATION_SET_PERCENT);

    select_random_data(training_data, used, total, train_count, data_array);
    select_random_data(test_data, used, total, test_count, data_array);
    select_random_data(validation_data, used, total, val_count, data_array);

    std::cout << "Training data size: " << training_data->size() << "." << std::endl;
    std::cout << "Test data size: " << test_data->size() << "." << std::endl;
    std::cout << "Validation data size: " << validation_data->size() << "." << std::endl;
}

void DataHandler::count_classes() {
    int count = 0;
    for (auto *dp : *data_array) {
        uint8_t label = dp->get_label();
        if (class_map.find(label) == class_map.end()) {
            class_map[label] = count;
            dp->set_enumerated_label(count);
            ++count;
        }
    }
    num_classes = count;
    std::cout << "Successfully extracted " << num_classes << " unique classes." << std::endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char *bytes) {
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

void DataHandler::normalize() {
    std::vector<double> mins, maxs;
    auto *first = data_array->at(0);

    for (auto val : *first->get_feature_vector()) {
        mins.push_back(val);
        maxs.push_back(val);
    }

    for (size_t i = 1; i < data_array->size(); ++i) {
        auto *dp = data_array->at(i);
        for (size_t j = 0; j < dp->get_feature_vector()->size(); ++j) {
            double val = dp->get_feature_vector()->at(j);
            mins[j] = std::min(mins[j], val);
            maxs[j] = std::max(maxs[j], val);
        }
    }

    for (auto *dp : *data_array) {
        dp->set_normalized_feature_vector(new std::vector<double>());
        dp->set_class_vector(num_classes);

        for (size_t j = 0; j < dp->get_feature_vector()->size(); ++j) {
            double norm;
            if (maxs[j] - mins[j] == 0) {
                norm = 0.0;
            } else {
                double raw = dp->get_feature_vector()->at(j);
                norm = (raw - mins[j]) / (maxs[j] - mins[j]);
            }
            dp->append_to_normalized_feature_vector(norm);
        }
    }
}

int DataHandler::get_class_count() const {
    return num_classes;
}

std::vector<DataPoint *> *DataHandler::get_training_data() const {
    return training_data;
}

std::vector<DataPoint *> *DataHandler::get_validation_data() const {
    return validation_data;
}

std::vector<DataPoint *> *DataHandler::get_test_data() const {
    return test_data;
}