#include "kmeans.hpp"
#include <cstdlib>      // for rand()
#include <cmath>        // for sqrt, pow
#include <limits>       // for numeric_limits
#include <unordered_set>

KMeans::KMeans(int k)
    : num_clusters(k),
      clusters(new std::vector<cluster_t *>()),
      used_indexes(new std::unordered_set<int>()) {}

/**
 * Initialize clusters by randomly selecting unique points from the training data.
 */
void KMeans::init_clusters() {
    while (clusters->size() < static_cast<size_t>(num_clusters)) {
        int index = rand() % training_set->size();
        while (used_indexes->find(index) != used_indexes->end()) {
            index = rand() % training_set->size();
        }
        clusters->push_back(new cluster_t(training_set->at(index)));
        used_indexes->insert(index);
    }
}

/**
 * Initialize clusters so that each unique class in training data gets one cluster.
 */
void KMeans::init_clusters_for_each_class() {
    std::unordered_set<int> classes_used;
    for (size_t i = 0; i < training_set->size(); ++i) {
        int label = training_set->at(i)->get_label();
        if (classes_used.find(label) == classes_used.end()) {
            clusters->push_back(new cluster_t(training_set->at(i)));
            classes_used.insert(label);
            used_indexes->insert(i);
        }
    }
}

/**
 * Train K-Means by assigning remaining points to the closest clusters.
 */
void KMeans::train() {
    while (used_indexes->size() < training_set->size()) {
        int index = rand() % training_set->size();
        while (used_indexes->find(index) != used_indexes->end()) {
            index = rand() % training_set->size();
        }

        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;

        for (size_t j = 0; j < clusters->size(); ++j) {
            double dist = euclidean_distance(*clusters->at(j)->centroid, training_set->at(index));
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = static_cast<int>(j);
            }
        }

        clusters->at(best_cluster)->add_to_cluster(training_set->at(index));
        used_indexes->insert(index);
    }
}

/**
 * Calculate Euclidean distance between centroid and a data point.
 */
double KMeans::euclidean_distance(const std::vector<double> &centroid, DataPoint *point) const {
    double dist = 0.0;
    const std::vector<double> *features = point->get_normalized_feature_vector();
    for (size_t i = 0; i < centroid.size(); ++i) {
        double diff = centroid[i] - features->at(i);
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

/**
 * Validate the model on the validation set, returning accuracy in [0, 100].
 */
double KMeans::validate() {
    double num_correct = 0.0;
    for (DataPoint *query_point : *validation_set) {
        double min_dist = std::numeric_limits<double>::max();
        int best = 0;

        for (size_t i = 0; i < clusters->size(); ++i) {
            double current_dist = euclidean_distance(*clusters->at(i)->centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best = static_cast<int>(i);
            }
        }

        if (clusters->at(best)->most_frequent_class == query_point->get_label()) {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / static_cast<double>(validation_set->size()));
}

/**
 * Test the model on the test set, returning accuracy in [0, 100].
 */
double KMeans::test() {
    double num_correct = 0.0;
    for (DataPoint *query_point : *test_set) {
        double min_dist = std::numeric_limits<double>::max();
        int best = 0;

        for (size_t i = 0; i < clusters->size(); ++i) {
            double current_dist = euclidean_distance(*clusters->at(i)->centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best = static_cast<int>(i);
            }
        }

        if (clusters->at(best)->most_frequent_class == query_point->get_label()) {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / static_cast<double>(test_set->size()));
}

/**
 * Get pointer to clusters.
 */
std::vector<cluster_t *> *KMeans::get_clusters() const {
    return clusters;
}