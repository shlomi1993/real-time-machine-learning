#include "data_point.hpp"

DataPoint::DataPoint()
    : feature_vector(new std::vector<uint8_t>),
      normalized_feature_vector(nullptr),
      label(0),
      enum_label(0),
      one_hot_encoded_label(nullptr),
      distance(0.0) {}

DataPoint::~DataPoint() {
    delete feature_vector;
    delete normalized_feature_vector;
    delete one_hot_encoded_label;
}

void DataPoint::set_feature_vector(std::vector<uint8_t> *vector) {
    delete feature_vector;
    feature_vector = vector;
}

void DataPoint::append_to_feature_vector(uint8_t value) {
    if (!feature_vector) {
        feature_vector = new std::vector<uint8_t>();
    }
    feature_vector->push_back(value);
}

void DataPoint::set_normalized_feature_vector(std::vector<double> *vector) {
    delete normalized_feature_vector;
    normalized_feature_vector = vector;
}

void DataPoint::append_to_normalized_feature_vector(double value) {
    if (!normalized_feature_vector) {
        normalized_feature_vector = new std::vector<double>();
    }
    normalized_feature_vector->push_back(value);
}

void DataPoint::set_class_vector(int num_classes) {
    delete one_hot_encoded_label;
    one_hot_encoded_label = new std::vector<int>(num_classes, 0);
    if (label < num_classes) {
        (*one_hot_encoded_label)[label] = 1;
    }
}

void DataPoint::set_label(uint8_t label) {
    this->label = label;
}

void DataPoint::set_enumerated_label(int enum_label) {
    this->enum_label = static_cast<uint8_t>(enum_label);
}

void DataPoint::set_distance(double dist) {
    this->distance = dist;
}

size_t DataPoint::get_feature_vector_size() const {
    return feature_vector ? feature_vector->size() : 0;
}

uint8_t DataPoint::get_label() const {
    return label;
}

uint8_t DataPoint::get_enumerated_label() const {
    return enum_label;
}

double DataPoint::get_distance() const {
    return distance;
}

std::vector<uint8_t>* DataPoint::get_feature_vector() {
    return feature_vector;
}

std::vector<double>* DataPoint::get_normalized_feature_vector() {
    return normalized_feature_vector;
}

std::vector<int> DataPoint::get_class_vector() {
    return one_hot_encoded_label ? *one_hot_encoded_label : std::vector<int>();
}