#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "data.hpp"

class Knn {
    int m_k;
    std::vector<Data *> *m_neighbors;
    std::vector<Data *> *m_training_data;
    std::vector<Data *> *m_test_data;
    std::vector<Data *> *m_validation_data;

public:
    Knn(int k);
    Knn() {}
    ~Knn() {}

    void find_k_nearest(Data *query_point);
    void set_training_data(std::vector<Data *> *vec);
    void set_test_data(std::vector<Data *> *vec);
    void set_validation_data(std::vector<Data *> *vec);
    void set_k(int k);

    int predict();
    double calculate_distance(Data *query_point, Data *input);
    double validate_performance();
    double test_performance();
};

#endif // __KNN_H