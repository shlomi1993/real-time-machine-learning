#pragma once

#include "data_set.hpp"

/**
 * @class KNN
 * @brief Implements the K-Nearest Neighbors (KNN) classification algorithm.
 *
 * Inherits from DataSet, providing access to the training, test, and validation datasets.
 */
class KNN : public DataSet {
private:
    int k;  ///< Number of nearest neighbors to consider.
    std::vector<DataPoint *> *neighbors;  ///< Pointer to the list of nearest neighbors.

public:
    /**
     * @brief Constructor with specified number of neighbors.
     * @param k Number of neighbors to use in prediction.
     */
    KNN(int k);

    /**
     * @brief Default constructor.
     */
    KNN() = default;

    /**
     * @brief Default destructor.
     */
    ~KNN() = default;

    /**
     * @brief Finds the k nearest neighbors of a given query point in the training data.
     * @param query_point The data point to classify.
     */
    void find_k_nearest(DataPoint *query_point);

    /**
     * @brief Sets the number of neighbors (k) to use in prediction.
     * @param k The new value of k.
     */
    void set_k(int k);

    /**
     * @brief Predicts the label of the current query point based on majority vote of k neighbors.
     * @return Predicted class label.
     */
    int predict();

    /**
     * @brief Calculates the Euclidean distance between two data points.
     * @param query_point The point to compare.
     * @param input A training data point.
     * @return Euclidean distance between query_point and input.
     */
    double calculate_distance(DataPoint *query_point, DataPoint *input);

    /**
     * @brief Evaluates classification accuracy on the validation dataset.
     * @return Accuracy as a percentage (0.0 - 100.0).
     */
    double validate_performance();

    /**
     * @brief Evaluates classification accuracy on the test dataset.
     * @return Accuracy as a percentage (0.0 - 100.0).
     */
    double test_performance();
};