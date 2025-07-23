#pragma once

#include <vector>
#include <unordered_set>
#include "data_set.hpp"
#include "cluster.hpp"

/**
 * @brief Implements the K-Means clustering algorithm for unsupervised learning.
 */
class KMeans : public DataSet {
private:
    /**
     * @brief Number of clusters to form.
     */
    int num_clusters;

    /**
     * @brief Pointer to the clusters (cluster_t structs).
     */
    std::vector<cluster_t *> *clusters;

    /**
     * @brief Set of indexes used to avoid duplicate initial clusters.
     */
    std::unordered_set<int> *used_indexes;

    /**
     * @brief Calculates Euclidean distance between a centroid and a data point.
     * @param centroid Vector of centroid values.
     * @param point Pointer to the data point.
     * @return The Euclidean distance.
     */
    double euclidean_distance(const std::vector<double> &centroid, DataPoint *point) const;

    /**
     * @brief Predicts the cluster index for a given data point.
     * @param point Pointer to the data point.
     * @return Cluster index.
     */
    int predict(DataPoint *point) const;

public:
    /**
     * @brief Constructs a KMeans object with the specified number of clusters.
     * @param k Number of clusters.
     */
    explicit KMeans(int k);

    /**
     * @brief Initializes centroids by randomly selecting points from the dataset.
     */
    void init_clusters();

    /**
     * @brief Initializes clusters so that each unique class in training data gets one cluster.
     */
    void init_clusters_for_each_class();

    /**
     * @brief Returns the pointer to clusters.
     * @return Pointer to vector of cluster_t pointers.
     */
    std::vector<cluster_t *> *get_clusters() const;

    /**
     * @brief Executes the K-Means clustering algorithm.
     */
    void train();

    /**
     * @brief Validates the model on the validation set.
     * @return Accuracy score in the range [0.0, 1.0].
     */
    double validate();

    /**
     * @brief Tests the model on the test set.
     * @return Accuracy score in the range [0.0, 1.0].
     */
    double test();
};