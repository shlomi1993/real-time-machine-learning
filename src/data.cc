#include "data.hpp"

Data::Data() : feature_vector(new std::vector<uint8_t>), label(0), enum_label(0), distance(0.0) {}

Data::~Data() {
    delete this->feature_vector;
}

void Data::set_feature_vector(std::vector<uint8_t> *feature_vector) {
    this->feature_vector = feature_vector;
}

void Data::set_normalized_feature_vector(std::vector<double> *feature_vector) {
    this->normalized_feature_vector = feature_vector;
}

void Data::append_to_feature_vector(uint8_t value) {
    this->feature_vector->push_back(value);
}

void Data::append_to_normalized_feature_vector(double value) {
    this->normalized_feature_vector->push_back(value);
}

void Data::set_label(uint8_t label) {
    this->label = label;
}

void Data::set_enumerated_label(int enum_label) {
    this->enum_label = enum_label;
}

void Data::set_distance(double distance) {
    this->distance = distance;
}

void Data::set_class_vector(int count) {
    class_vector = new std::vector<int>();
    for (int i = 0; i < count; ++i) {
        class_vector->push_back((i == this->label) ? 1 : 0);
    }
}

size_t Data::get_feature_vector_size() const {
    return this->feature_vector->size();
}

uint8_t Data::get_label() const {
    return this->label;
}

uint8_t Data::get_enumerated_label() const {
    return this->enum_label;
}

std::vector<uint8_t> *Data::get_feature_vector() {
    return this->feature_vector;
}

std::vector<double> *Data::get_normalized_feature_vector() {
    return this->normalized_feature_vector;
}

std::vector<int> Data::get_class_vector() {
    return *(this->class_vector);
}

double Data::get_distance() const {
    return this->distance;
}