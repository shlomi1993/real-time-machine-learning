#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"

class Data {
    std::vector<uint8_t> *m_feature_vector; // No class at end.
    uint8_t m_label;
    int m_enum_label; // A -> 1, B -> 2
    double m_distance;

  public:
    Data();
    ~Data() {}
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerated_label();
    void set_distance(double distance);

    std::vector<uint8_t> *get_feature_vector();
    double get_distance();
};


#endif