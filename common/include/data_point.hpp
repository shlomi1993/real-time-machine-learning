#pragma once

#include <vector>
#include <cstdint>
#include <memory>

/**
 * @brief Represents a single data point (e.g., an image and its label).
 */
class DataPoint {
private:
    // Raw feature values (e.g., pixel intensities)
    std::vector<uint8_t> *feature_vector = nullptr;

    // Normalized feature values in range [0, 1]
    std::vector<double> *normalized_feature_vector = nullptr;

    // Original label from dataset
    uint8_t label = 0;

    // Enumerated label (e.g., index in class map)
    uint8_t enum_label = 0;

    // One-hot encoded class label
    std::vector<int> *one_hot_encoded_label = nullptr;

    // Distance from this point to a cluster centroid (used in KMeans, etc.)
    double distance = 0.0;

public:
    /**
     * @brief Constructs a new DataPoint object.
     */
    DataPoint();

    /**
     * @brief Destroys the DataPoint object and releases owned memory.
     */
    ~DataPoint();

    /**
     * @brief Sets the feature vector.
     * @param vector Pointer to a vector of raw features.
     */
    void set_feature_vector(std::vector<uint8_t> *vector);

    /**
     * @brief Appends a raw feature value.
     * @param value Feature to add.
     */
    void append_to_feature_vector(uint8_t value);

    /**
     * @brief Sets the normalized feature vector.
     * @param vector Pointer to a vector of normalized features.
     */
    void set_normalized_feature_vector(std::vector<double> *vector);

    /**
     * @brief Appends a normalized feature value.
     * @param value Normalized feature to add.
     */
    void append_to_normalized_feature_vector(double value);

    /**
     * @brief Initializes the one-hot class vector with the number of classes.
     * @param num_classes Total number of unique classes.
     */
    void set_class_vector(int num_classes);

    /**
     * @brief Sets the raw class label.
     * @param label Raw label.
     */
    void set_label(uint8_t label);

    /**
     * @brief Sets the enumerated label (mapped to class index).
     * @param enum_label Enumerated label.
     */
    void set_enumerated_label(int enum_label);

    /**
     * @brief Sets the distance (e.g., to a cluster center).
     * @param dist Distance value.
     */
    void set_distance(double dist);

    /**
     * @brief Returns the size of the feature vector.
     * @return Number of raw features.
     */
    size_t get_feature_vector_size() const;

    /**
     * @brief Returns the original label.
     * @return Label value.
     */
    uint8_t get_label() const;

    /**
     * @brief Returns the enumerated label.
     * @return Enumerated label value.
     */
    uint8_t get_enumerated_label() const;

    /**
     * @brief Returns the distance to a reference point.
     * @return Distance value.
     */
    double get_distance() const;

    /**
     * @brief Returns a pointer to the raw feature vector.
     * @return Raw feature vector.
     */
    std::vector<uint8_t>* get_feature_vector();

    /**
     * @brief Returns a pointer to the normalized feature vector.
     * @return Normalized feature vector.
     */
    std::vector<double>* get_normalized_feature_vector();

    /**
     * @brief Returns a copy of the one-hot class vector.
     * @return One-hot encoded class vector.
     */
    std::vector<int> get_class_vector();
};