#pragma once

#include "common.hpp"

class Knn : public CommonData {
    int m_k;
    std::vector<Data *> *m_neighbors;

public:
    Knn(int k);
    Knn() {}
    ~Knn() {}

    void find_k_nearest(Data *query_point);
    void set_k(int k);
    int predict();
    double calculate_distance(Data *query_point, Data *input);
    double validate_performance();
    double test_performance();
};