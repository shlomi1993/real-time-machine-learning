#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"
#include "knn.hpp"

Knn::Knn(int k) { set_k(k); }

void Knn::set_training_data(std::vector<Data *> *vec) { m_training_data = vec; }

void Knn::set_test_data(std::vector<Data *> *vec) { m_test_data = vec; }

void Knn::set_validation_data(std::vector<Data *> *vec) { m_validation_data = vec; }

void Knn::set_k(int k) {
    std::cout << "Set K to " << k << "." << std::endl;
    m_k = k;
}

double Knn::calculate_distance(Data *query_point, Data *input) {
    double distance = 0.0;
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
        std::cerr << "Feature vectors of different sizes." << std::endl;
        exit(1);
    }
#ifdef EUCLID
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); ++i) {
        auto diff = query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i);
        distance += pow(diff, 2);
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
    // IMPLEMENT MANHATTAN DISTANCE
#endif
    return distance;
}

void Knn::find_k_nearest(Data *query_point) {
    m_neighbors = new std::vector<Data *>;
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int index = 0;

    // First pass (i = 0) - calculate distances
    for (int j = 0; j < m_training_data->size(); ++j) {
        double distance = calculate_distance(query_point, m_training_data->at(j));
        m_training_data->at(j)->set_distance(distance);
        if (distance < min) {
            min = distance;
            index = j;
        }
    }
    m_neighbors->push_back(m_training_data->at(index));
    previous_min = min;
    min = std::numeric_limits<double>::max();

    // Remaining passes (i > 0)
    for (int i = 1; i < m_k; ++i) {
        for (int j = 0; j < m_training_data->size(); ++j) {
            double distance = m_training_data->at(j)->get_distance();
            if (distance > previous_min && distance < min) {
                min = distance;
                index = j;
            }
        }
        m_neighbors->push_back(m_training_data->at(index));
        previous_min = min;
        min = std::numeric_limits<double>::max();
    }
}

int Knn::predict() {
    std::map<uint8_t, int> class_freq;
    for (int i = 0; i < m_neighbors->size(); ++i) {
        auto label = m_neighbors->at(i)->get_label();
        if (class_freq.find(label) == class_freq.end()) {
            class_freq[label] = 1;
        } else {
            class_freq[label]++;
        }
    }

    int best = 0;
    int max = 0;
    for (auto kv : class_freq) {
        if (kv.second > max) {
            best = kv.first;
            max = kv.second;
        }
    }
    m_neighbors->clear();
    return best;
}

double Knn::validate_performance() {
    int count = 0;
    int data_index = 0;
    for (Data *query_point: *m_validation_data) {
        find_k_nearest(query_point);
        int predicted_label = predict();
        if (predicted_label == query_point->get_label()) {
            ++count;
        }
        ++data_index;
        std::cout << "Current performance is " << ((double) count * 100.0) / ((double) data_index) << "%" << std::endl;
    }
    auto validation_performance = ((double) count * 100.0) / ((double) m_validation_data->size());
    std::cout << "Validation performance is " << validation_performance << "%" << std::endl;
    return validation_performance;

}

double Knn::test_performance() {
    int count = 0;
    for (Data *query_point: *m_test_data) {
        find_k_nearest(query_point);
        int predicted_label = predict();
        if (predicted_label == query_point->get_label()) {
            ++count;
        }
    }
    auto test_performance = ((double) count * 100.0) / ((double) m_test_data->size());
    std::cout << "Test performance is " << test_performance << "%" << std::endl;
    return test_performance;

}

// Unittest - KNN
int main() {
    DataHandler *dh = new DataHandler();
    dh->read_feature_vector("../dataset/train-images-idx3-ubyte");
    dh->read_feature_labels("../dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    Knn *knn = new Knn();
    knn->set_training_data(dh->get_training_data());
    knn->set_test_data(dh->get_test_data());
    knn->set_validation_data(dh->get_validation_data());

    double performance = -1;
    double best_performances = -1;
    int best_k = -1;
    for (int i = 1; i < 5; ++i) {
        knn->set_k(i);
        performance = knn->validate_performance();
        if (performance > best_performances) {
            best_performances = performance;
            best_k = i;
        }
    }

    std::cout << "Best k value is " << best_k << ", with validation performance " << best_performances << "%" << std::endl;
    delete dh;
    delete knn;
    return 0;
}