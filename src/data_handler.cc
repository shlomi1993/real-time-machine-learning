#include <iostream>
#include "data_handler.hpp"

DataHandler::DataHandler() noexcept {
    data_array = new std::vector<Data *>;
    training_data = new std::vector<Data *>;
    test_data = new std::vector<Data *>;
    validation_data = new std::vector<Data *>;
}

DataHandler::~DataHandler() {
    delete data_array;
    delete training_data;
    delete test_data;
    delete validation_data;
}

void DataHandler::read_csv(std::string path, std::string delim) {
    this->n_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line; // Holds each line
    while (std::getline(data_file, line)) {
        if (line.length() == 0) {
            continue;
        }
        Data *d = new Data();
        d->set_normalized_feature_vector(new std::vector<double>());
        size_t pos = 0;
        std::string token; // Value in between delimiter
        while ((pos = line.find(delim)) != std::string::npos) {
            token = line.substr(0, pos);
            d->append_to_normalized_feature_vector(std::stod(token));
            line.erase(0, pos + delim.length());
        }
        if (this->str_class_map.find(line) != this->str_class_map.end()) {
            d->set_label(this->str_class_map[line]);
        } else {
            this->str_class_map[line] = this->n_classes;
            d->set_label(this->str_class_map[line]);
            this->n_classes++;
        }
        this->data_array->push_back(d);
        this->feature_vector_size = this->data_array->at(0)->get_normalized_feature_vector()->size();
    }
}

void DataHandler::read_csv(std::string path) {
    return this->read_csv(path, ",");
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
            this->data_array->push_back(feature_vector);
        }
        std::cout << "Successfully read and stored " << this->data_array->size() << " feature vectors." << std::endl;
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
                this->data_array->at(i)->set_label(element[0]);
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

    auto size = this->data_array->size();
    int train_size = static_cast<int>(size * this->TRAIN_SET_PERCENT);
    int test_size = static_cast<int>(size * this->TEST_SET_PERCENT);
    int validation_size = static_cast<int>(size * this->VALIDATION_SET_PERCENT);

    select_random_data(this->training_data, used_indexes, size, train_size, this->data_array);
    select_random_data(this->test_data, used_indexes, size, test_size, this->data_array);
    select_random_data(this->validation_data, used_indexes, size, validation_size, this->data_array);


    std::cout << "Training data size: " << this->training_data->size() << "." << std::endl;
    std::cout << "Test data size: " << this->test_data->size() << "." << std::endl;
    std::cout << "Validation data size: " << this->validation_data->size() << "." << std::endl;
}

void DataHandler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < this->data_array->size(); ++i) {
        if (class_map.find(this->data_array->at(i)->get_label()) == this->class_map.end()) {
            this->class_map[this->data_array->at(i)->get_label()] = count;
            this->data_array->at(i)->set_enumerated_label(count);
            ++count;
        }
    }
    this->n_classes = count;
    std::cout << "Successfully extracted " << count << " unique classes." << std::endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char *bytes) {
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3] << 0));
}

void DataHandler::normalize() {
    std::vector<double> mins, maxs;

    // Fill min and max lists
    Data *d = this->data_array->at(0);
    for (auto val : *d->get_feature_vector()) {
        mins.push_back(val);
        maxs.push_back(val);
    }

    for (int i = 1; i < this->data_array->size(); i++) {
        d = this->data_array->at(i);
        for (int j = 0; j < d->get_feature_vector()->size(); j++) {
            double value = (double) d->get_feature_vector()->at(j);
            if (value < mins.at(j)) mins[j] = value;
            if (value > maxs.at(j)) maxs[j] = value;
        }
    }

    // Normalize data array
    for (int i = 0; i < this->data_array->size(); i++) {
        this->data_array->at(i)->set_normalized_feature_vector(new std::vector<double>());
        this->data_array->at(i)->set_class_vector(n_classes);
        for (int j = 0; j < this->data_array->at(i)->get_feature_vector()->size(); j++) {
            if (maxs[j] - mins[j] == 0) {
                this->data_array->at(i)->append_to_feature_vector(0.0);
            } else {
                auto data_point = this->data_array->at(i)->get_feature_vector()->at(j);
                auto normalized_data_point = (double) (data_point - mins[j]) / (maxs[j] - mins[j]);
                this->data_array->at(i)->append_to_feature_vector(normalized_data_point);
            }
        }
    }
}

int DataHandler::get_class_counts() {
    return this->n_classes;
}

std::vector<Data *> *DataHandler::get_training_data() {
    return this->training_data;
}

std::vector<Data *> *DataHandler::get_validation_data() {
    return this->validation_data;
}

std::vector<Data *> *DataHandler::get_test_data() {
    return this->test_data;
}