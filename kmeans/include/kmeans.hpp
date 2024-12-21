#pragma once

#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "common.hpp"
#include "data_handler.hpp"

typedef struct Cluster {
    std::vector<double> *m_centroid;
    std::vector<Data *> *m_cluster_points;
    std::map<int, int> m_class_counts;
    int m_most_frequent_class;

    Cluster(Data *initial_point) {
        m_centroid = new std::vector<double>;
        m_cluster_points = new std::vector<Data *>;
        for (auto value : *(initial_point->get_feature_vector())) {
            m_centroid->push_back(value);
        }
        m_cluster_points->push_back(initial_point);
        m_class_counts[initial_point->get_label()] = 1;
        m_most_frequent_class = initial_point->get_label();
    }

    void add_to_cluster(Data *point) {
        int previous_size = m_cluster_points->size();
        m_cluster_points->push_back(point);
        for (int i = 0; i < m_centroid->size() - 1; ++i) {
            double value = m_centroid->at(i);
            value *= previous_size;
            value += point->get_feature_vector()->at(i);
            value /= (double) m_cluster_points->size();
            m_centroid->at(i) = value;
        }
        if (m_class_counts.find(point->get_label()) == m_class_counts.end()) {
            m_class_counts[point->get_label()] = 1;
        } else {
            m_class_counts[point->get_label()]++;
        }
        set_most_frequent_class();
    }

    void set_most_frequent_class() {
        int best_class;
        int freq = 0;
        for (auto kv : m_class_counts) {
            best_class = kv.first;
            freq = kv.second;
        }
        m_most_frequent_class = best_class;
    }

    ~Cluster() {
        delete m_centroid;
        delete m_cluster_points;
    }
} Cluster;

class KMeans : public CommonData {
    int m_n_clusters;
    std::vector<Cluster *> *m_clusters;
    std::unordered_set<int> *m_used_indexes;

public:
    KMeans(int k);
    void init_clusters();
    void init_clusters_for_each_class();
    double euclidean_distance(std::vector<double> *centroids, Data *point);
    void train();
    double validate();
    double test();
};