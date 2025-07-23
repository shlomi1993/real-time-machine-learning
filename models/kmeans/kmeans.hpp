#pragma once

#include <vector>
#include <memory>
#include <unordered_set>
#include <map>
#include "../include/common.hpp"
#include "../include/data_handler.hpp"
#include "cluster.hpp"

/**
 * @brief Implements the K-Means clustering algorithm for unsupervised learning.
 */
class KMeans : public CommonData {
private:
    /**
     * @brief Number of clusters to form.
     */
    int m_clusters;

    /**
     * @brief Pointer to the input dataset to be clustered.
     */
    std::vector<Data*>* m_data = nullptr;

    /**
     * @brief List of centroids, one for each cluster.
     */
    std::vector<Data*> m_centroids;

    /**
     * @brief Calculates Euclidean distance between a centroid and a data point.
     * @param centroid Vector of centroid values.
     * @param point Pointer to the data point.
     * @return The Euclidean distance.
     */
    double euclidean_distance(const std::vector<double>& centroid, Data* point) const;

    /**
     * @brief Predicts the cluster index for a given data point.
     * @param point Pointer to the data point.
     * @return Cluster index.
     */
    int predict(Data* point) const;

public:
    /**
     * @brief Constructs a KMeans object with the specified number of clusters.
     * @param k Number of clusters.
     */
    explicit KMeans(int k);

    /**
     * @brief Sets the dataset to be used for clustering.
     * @param input_data Pointer to the vector of data points.
     */
    void set_data(std::vector<Data*>* input_data);

    /**
     * @brief Initializes centroids by randomly selecting points from the dataset.
     */
    void init_clusters();

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
