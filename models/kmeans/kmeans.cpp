#include <cstdlib>
#include <limits>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include "kmeans.hpp"
#include "../include/common.hpp"  // for validation_data and test_data

/**
 * @brief Constructs a KMeans object with the specified number of clusters.
 * @param k Number of clusters.
 */
KMeans::KMeans(int k) : m_clusters(k) {}

/**
 * @brief Sets the dataset to be used for clustering.
 * @param input_data Pointer to the vector of data points.
 */
void KMeans::set_data(std::vector<Data*>* input_data) {
    m_data = input_data;
}

/**
 * @brief Initializes centroids by selecting k unique random data points from the dataset.
 */
void KMeans::init_clusters() {
    std::unordered_set<int> selected_indices;

    while (static_cast<int>(m_centroids.size()) < m_clusters) {
        int idx = rand() % m_data->size();
        if (selected_indices.insert(idx).second) {
            m_centroids.push_back((*m_data)[idx]);
        }
    }
}

/**
 * @brief Trains the KMeans model using the training dataset.
 */
void KMeans::train() {
    std::vector<Cluster> clusters;
    for (Data* centroid_point : m_centroids) {
        clusters.emplace_back(centroid_point);
    }

    for (Data* point : *m_data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster_idx = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            double dist = euclidean_distance(clusters[i].centroid, point);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster_idx = static_cast<int>(i);
            }
        }

        clusters[best_cluster_idx].add_to_cluster(point);
    }

    m_centroids.clear();
    for (const auto& cluster : clusters) {
        Data* dummy = new Data();
        dummy->set_normalized_feature_vector(new std::vector<double>(cluster.centroid));
        dummy->set_label(cluster.most_frequent_class);
        m_centroids.push_back(dummy);
    }
}

/**
 * @brief Computes the Euclidean distance between a centroid and a data point.
 * @param centroid Reference to centroid vector.
 * @param point Pointer to the data point.
 * @return Euclidean distance as double.
 */
double KMeans::euclidean_distance(const std::vector<double>& centroid, Data* point) const {
    const auto* features = point->get_normalized_feature_vector();
    double sum = 0.0;
    for (size_t i = 0; i < centroid.size(); ++i) {
        double diff = centroid[i] - (*features)[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Validates the model using the validation dataset.
 * @return Classification accuracy (0.0 - 100.0).
 */
double KMeans::validate() {
    double correct = 0.0;
    for (Data* point : *validation_data) {
        int predicted = predict(point);
        if (predicted == point->get_label()) {
            ++correct;
        }
    }
    return 100.0 * correct / validation_data->size();
}

/**
 * @brief Tests the model using the test dataset.
 * @return Classification accuracy (0.0 - 100.0).
 */
double KMeans::test() {
    double correct = 0.0;
    for (Data* point : *test_data) {
        int predicted = predict(point);
        if (predicted == point->get_label()) {
            ++correct;
        }
    }
    return 100.0 * correct / test_data->size();
}

/**
 * @brief Predicts the class label for a given data point using nearest centroid.
 * @param point Pointer to the data point.
 * @return Predicted label.
 */
int KMeans::predict(Data* point) const {
    double min_dist = std::numeric_limits<double>::max();
    int best_label = -1;

    for (Data* centroid : m_centroids) {
        double dist = euclidean_distance(*centroid->get_normalized_feature_vector(), point);
        if (dist < min_dist) {
            min_dist = dist;
            best_label = centroid->get_label();
        }
    }

    return best_label;
}
