#include <cmath>  // for isnan
#include "cluster.hpp"

Cluster::Cluster(DataPoint *initial_point)
    : centroid(new std::vector<double>()),
      cluster_points(new std::vector<DataPoint *>()) {
    const std::vector<double> *features = initial_point->get_normalized_feature_vector();
    for (double val : *features) {
        if (std::isnan(val)) {
            centroid->push_back(0.0);
        } else {
            centroid->push_back(val);
        }
    }
    cluster_points->push_back(initial_point);

    int label = initial_point->get_label();
    class_counts[label] = 1;
    most_frequent_class = label;
}

void Cluster::add_to_cluster(DataPoint *point) {
    int previous_size = cluster_points->size();
    cluster_points->push_back(point);

    const std::vector<double> *features = point->get_normalized_feature_vector();
    for (size_t i = 0; i < centroid->size(); ++i) {
        double val = centroid->at(i);
        val *= previous_size;
        val += features->at(i);
        val /= static_cast<double>(cluster_points->size());
        centroid->at(i) = val;
    }

    int label = point->get_label();
    if (class_counts.find(label) == class_counts.end()) {
        class_counts[label] = 1;
    } else {
        class_counts[label]++;
    }

    update_most_frequent_class();
}

void Cluster::update_most_frequent_class() {
    int best_class = -1;
    int max_freq = 0;
    for (const auto &kv : class_counts) {
        if (kv.second > max_freq) {
            max_freq = kv.second;
            best_class = kv.first;
        }
    }
    most_frequent_class = best_class;
}