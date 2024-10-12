#pragma once

#include <vector>
#include <cstdint>
#include <memory>

class Data {
    std::vector<uint8_t> *m_feature_vector; // No class at end.
    uint8_t m_label;
    uint8_t m_enum_label; // A -> 1, B -> 2
    double m_distance;

public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> *feature_vector);
    void append_to_feature_vector(uint8_t value);
    void set_label(uint8_t label);
    void set_enumerated_label(int enum_label);

    size_t get_feature_vector_size() const;
    uint8_t get_label() const;
    uint8_t get_enumerated_label() const;
    void set_distance(double distance);

    std::vector<uint8_t> *get_feature_vector();
    double get_distance() const;
};