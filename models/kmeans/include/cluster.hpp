#pragma once

#include <vector>
#include <map>
#include "data_handler.hpp"

/**
 * @brief Represents a cluster in the K-Means algorithm.
 */
typedef struct Cluster {
    std::vector<double> *centroid;
    std::vector<DataPoint *> *cluster_points;
    std::map<int, int> class_counts;
    int most_frequent_class;

    /**
     * @brief Construct a cluster initialized with a single data point.
     * @param initial_point Pointer to the initial data point.
     */
    explicit Cluster(DataPoint *initial_point);

    /**
     * @brief Add a data point to the cluster and update its state.
     * @param point Pointer to the data point.
     */
    void add_to_cluster(DataPoint *point);

private:
    /**
     * @brief Update the most frequent class in the cluster.
     */
    void update_most_frequent_class();
} cluster_t;