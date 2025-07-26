#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include <cstdint>
#include "knn.hpp"

// Constructor that sets the value of k
KNN::KNN(int k) {
    set_k(k);
}

// Set the number of nearest neighbors (k)
void KNN::set_k(int k) {
    if (k <= 0) {
        std::cerr << "Error: k must be positive." << std::endl;
        exit(1);
    }

    std::cout << "Set k to " << k << "." << std::endl;
    this->k = k;
}

// Calculate the distance between two data points using Euclidean or Manhattan distance
double KNN::calculate_distance(DataPoint *query_point, DataPoint *input) {
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
        std::cerr << "Error: Feature vectors have different sizes." << std::endl;
        exit(1);
    }

    double distance = 0.0;

    // Compute Euclidean distance
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); ++i) {
        double diff = query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i);
        distance += diff * diff;
    }
    distance = std::sqrt(distance);

    // // Alternatively, compute Manhattan distance
    // for (unsigned i = 0; i < query_point->get_feature_vector_size(); ++i) {
    //     distance += std::abs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
    // }

    return distance;
}

// Find the k nearest neighbors in the training set for the given query point
void KNN::find_k_nearest(DataPoint *query_point) {
    this->neighbors = new std::vector<DataPoint *>;

    // Step 1: compute distance for all training points and find the closest one
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int index = 0;

    for (size_t j = 0; j < this->training_set->size(); ++j) {
        double distance = this->calculate_distance(query_point, this->training_set->at(j));
        this->training_set->at(j)->set_distance(distance);

        if (distance < min) {
            min = distance;
            index = j;
        }
    }

    // Add the closest point to neighbors
    this->neighbors->push_back(this->training_set->at(index));
    previous_min = min;
    min = std::numeric_limits<double>::max();

    // Step 2: find the next closest k-1 neighbors
    for (int i = 1; i < k; ++i) {
        for (size_t j = 0; j < this->training_set->size(); ++j) {
            double distance = this->training_set->at(j)->get_distance();

            if (distance > previous_min && distance < min) {
                min = distance;
                index = j;
            }
        }
        this->neighbors->push_back(this->training_set->at(index));
        previous_min = min;
        min = std::numeric_limits<double>::max();
    }
}

// Predict the class label of the current query point using majority vote
int KNN::predict() {
    std::map<uint8_t, int> class_freq;

    // Count class frequency among neighbors
    for (DataPoint *neighbor : *this->neighbors) {
        class_freq[neighbor->get_label()]++;
    }

    // Find the label with the highest vote
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

// Evaluate the model’s accuracy on a given dataset (test or validation)
double evaluate_performance(KNN &knn, std::vector<DataPoint *> *data_set, const std::string &set_name) {
    int correct_count = 0;
    int total = 0;
    double performance = 0.0;

    for (DataPoint *query_point : *data_set) {
        knn.find_k_nearest(query_point);
        int predicted_label = knn.predict();

        if (predicted_label == query_point->get_label()) {
            ++correct_count;
        }

        ++total;
        performance = (static_cast<double>(correct_count) * 100.0) / total;

        if (correct_count % 10 == 0) {
            std::cout << "Current " << set_name << " performance: " << performance << "%" << std::endl;
        }
    }

    performance = (static_cast<double>(correct_count) * 100.0) / data_set->size();
    std::cout << "Final " << set_name << " performance: " << performance << "%" << std::endl;
    return performance;
}

// Wrapper for evaluating validation set accuracy
double KNN::validate_performance() {
    return evaluate_performance(*this, this->validation_set, "validation");
}

// Wrapper for evaluating test set accuracy
double KNN::test_performance() {
    return evaluate_performance(*this, this->test_set, "test");
}