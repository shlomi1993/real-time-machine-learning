#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <cstdint>
#include <memory>

class Data {
    std::vector<uint8_t> m_feature_vector;
    uint8_t m_label;
    int m_enum_label; // A -> 1, B -> 2
    double m_distance;

public:
    Data();
    void set_feature_vector(const std::vector<uint8_t>& feature_vector);
    void append_to_feature_vector(uint8_t value);
    void set_label(uint8_t label);
    void set_enumerated_label(int enum_label);
    void set_distance(double distance);

    int get_feature_vector_size() const;
    uint8_t get_label() const;
    int get_enumerated_label() const;
    const std::vector<uint8_t>& get_feature_vector() const;
    double get_distance() const;
};

#endif // DATA_HPP