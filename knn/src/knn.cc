#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"
#include "knn.hpp"

Knn::Knn(int k) { set_k(k); }

void Knn::set_k(int k) {
    if (k <= 0) {
        std::cerr << "Error: k must be positive." << std::endl;
        exit(1);
    }

    std::cout << "Set K to " << k << "." << std::endl;
    m_k = k;
}

double Knn::calculate_distance(Data *query_point, Data *input) {
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
        std::cerr << "Feature vectors of different sizes." << std::endl;
        exit(1);
    }
    double distance = 0.0;
#ifdef EUCLID
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); ++i) {
        double diff = query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i);
        distance += diff * diff;
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); ++i) {
        distance += std::abs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
    }
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
    for (Data *neighbor : *m_neighbors) {
        class_freq[neighbor->get_label()]++;
    }

    uint8_t best_label = 0;
    int max_count = 0;
    for (const auto &kv : class_freq) {
        if (kv.second > max_count) {
            best_label = kv.first;
            max_count = kv.second;
        }
    }

    return best_label;
}

// Standalone function to evaluate performance
double evaluate_performance(Knn &knn, std::vector<Data *> *data_set, const std::string &set_name) {
    int count = 0;
    int data_index = 0;
    double performance;

    for (Data *query_point : *data_set) {
        knn.find_k_nearest(query_point);
        int predicted_label = knn.predict();
        if (predicted_label == query_point->get_label()) {
            ++count;
        }
        ++data_index;
        performance = (static_cast<double>(count) * 100.0) / static_cast<double>(data_index);
        std::cout << "Current " << set_name << " performance: " << performance << "%" << std::endl;
    }

    performance = (static_cast<double>(count) * 100.0) / static_cast<double>(data_set->size());
    std::cout << "Final " << set_name << " performance: " << performance << "%" << std::endl;
    return performance;
}

double Knn::validate_performance() {
    return evaluate_performance(*this, m_validation_data, "validation");
}

double Knn::test_performance() {
    return evaluate_performance(*this, m_test_data, "test");
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