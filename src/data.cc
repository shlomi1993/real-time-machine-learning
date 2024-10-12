#include "data.hpp"

Data::Data() { m_feature_vector = new std::vector<uint8_t>; }

void Data::set_feature_vector(std::vector<uint8_t> *feature_vector) { m_feature_vector = feature_vector; }

void Data::append_to_feature_vector(uint8_t value) { m_feature_vector->push_back(value); }

void Data::set_label(uint8_t label) { m_label = label; }

void Data::set_enumerated_label(int enum_label) { m_enum_label = enum_label; }

void Data::set_distance(double distance) { m_distance = distance; }

int Data::get_feature_vector_size() { return m_feature_vector->size(); }

uint8_t Data::get_label() { return m_label; }

uint8_t Data::get_enumerated_label() { return m_enum_label; }

std::vector<uint8_t> *Data::get_feature_vector() { return m_feature_vector; }

double Data::get_distance() { return m_distance; }