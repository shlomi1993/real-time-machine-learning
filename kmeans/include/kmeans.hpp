#pragma once

#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "common.hpp"
#include "data_handler.hpp"

typedef struct Cluster {
    std::vector<double> *centroid;
    std::vector<Data *> *cluster_points;
    std::map<int, int> class_counts;
    int most_frequent_class;

    Cluster(Data *initial_point) {
        this->centroid = new std::vector<double>;
        this->cluster_points = new std::vector<Data *>;
        for (auto value : *(initial_point->get_feature_vector())) {
            this->centroid->push_back(value);
        }
        this->cluster_points->push_back(initial_point);
        this->class_counts[initial_point->get_label()] = 1;
        this->most_frequent_class = initial_point->get_label();
    }

    void add_to_cluster(Data *point) {
        int previous_size = this->cluster_points->size();
        this->cluster_points->push_back(point);
        for (int i = 0; i < this->centroid->size() - 1; ++i) {
            double value = this->centroid->at(i);
            value *= previous_size;
            value += point->get_feature_vector()->at(i);
            value /= (double) this->cluster_points->size();
            this->centroid->at(i) = value;
        }
        if (this->class_counts.find(point->get_label()) == this->class_counts.end()) {
            this->class_counts[point->get_label()] = 1;
        } else {
            this->class_counts[point->get_label()]++;
        }
        set_most_frequent_class();
    }

    void set_most_frequent_class() {
        int best_class;
        int freq = 0;
        for (auto kv : this->class_counts) {
            best_class = kv.first;
            freq = kv.second;
        }
        this->most_frequent_class = best_class;
    }

    ~Cluster() {
        delete this->centroid;
        delete this->cluster_points;
    }
} Cluster;

class KMeans : public CommonData {
    int n_clusters;
    std::vector<Cluster *> *clusters;
    std::unordered_set<int> *used_indexes;

public:
    KMeans(int k);
    void init_clusters();
    void init_clusters_for_each_class();
    double euclidean_distance(std::vector<double> *centroids, Data *point);
    void train();
    double validate();
    double test();
};