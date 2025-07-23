#include "cluster.hpp"
#include <limits>
#include <cmath>

/**
 * @brief Constructs a cluster initialized with a single data point.
 *
 * Initializes the centroid to the normalized feature vector of the point,
 * adds the point to the cluster, and sets the initial class count.
 *
 * @param initial_point Pointer to the initial data point.
 */
Cluster::Cluster(Data* initial_point) {
    centroid = *initial_point->get_normalized_feature_vector();
    cluster_points.push_back(initial_point);
    class_counts[initial_point->get_label()] = 1;
    most_frequent_class = initial_point->get_label();
}

/**
 * @brief Adds a new data point to the cluster and updates class statistics.
 *
 * The centroid is recomputed as the mean of all normalized vectors in the cluster.
 * The most frequent class is updated accordingly.
 *
 * @param point Pointer to the data point to add.
 */
void Cluster::add_to_cluster(Data* point) {
    cluster_points.push_back(point);
    ++class_counts[point->get_label()];

    const std::vector<double>* point_vec = point->get_normalized_feature_vector();
    for (size_t i = 0; i < centroid.size(); ++i) {
        centroid[i] = (centroid[i] * (cluster_points.size() - 1) + (*point_vec)[i]) / cluster_points.size();
    }

    update_most_frequent_class();
}

/**
 * @brief Updates the most frequent class label among the cluster's points.
 */
void Cluster::update_most_frequent_class() {
    int max_count = 0;
    for (const auto& kv : class_counts) {
        if (kv.second > max_count) {
            max_count = kv.second;
            most_frequent_class = kv.first;
        }
    }
}