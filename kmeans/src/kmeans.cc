#include "kmeans.hpp"

KMeans::KMeans(int k) {
    m_n_clusters = k;
    m_clusters = new std::vector<Cluster *>();
    m_used_indexes = new std::unordered_set<int>();
}

void KMeans::init_clusters() {
    std::cout << "Initialize clusters" << std::endl;
    for (int i = 0; i < m_n_clusters; ++i) {
        int index = rand() % m_training_data->size();
        while (m_used_indexes->find(index) != m_used_indexes->end()) {
            index = rand() % m_training_data->size();
        }
        m_clusters->push_back(new Cluster(m_training_data->at(index)));
        m_used_indexes->insert(index);
    }
}

void KMeans::init_clusters_for_each_class() {
    std::cout << "Initialize cluster for each class" << std::endl;
    std::unordered_set<int> classes_used;
    for (int i = 0; i < m_training_data->size(); ++i) {
        if (classes_used.find(m_training_data->at(i)->get_label()) == classes_used.end()) {
            std::cout << "Initialize cluster for '" << m_training_data->at(i)->get_label() << "'" << std::endl;
            m_clusters->push_back(new Cluster(m_training_data->at(i)));
            classes_used.insert(m_training_data->at(i)->get_label());
            m_used_indexes->insert(i);
        }
    }
}

double KMeans::euclidean_distance(std::vector<double> *centroids, Data *point) {
    double distance = 0.0;
    double diff;
    for (int i = 0; i < centroids->size(); ++i) {
        diff = centroids->at(i) - point->get_feature_vector()->at(i);
        distance += diff * diff;
    }
    return sqrt(distance);
}

void KMeans::train() {
    std::cout << "Training... ";
    int index;
    do {
        index = rand() % m_training_data->size();
    } while (m_used_indexes->find(index) != m_used_indexes->end());
    double min_dist = std::numeric_limits<double>::max();
    int best_cluster = 0;
    for (int i = 0; i < m_clusters->size(); ++i) {
        double current_dist = euclidean_distance(m_clusters->at(i)->m_centroid, m_training_data->at(index));
        if (current_dist < min_dist) {
            min_dist = current_dist;
        }
    }
    m_clusters->at(best_cluster)->add_to_cluster(m_training_data->at(index));
    m_used_indexes->insert(index);
    std::cout << "training completed" << std::endl;
}

double KMeans::validate() {
    std::cout << "Validating... ";
    double num_correct = 0.0;
    for (auto query_point: *m_validation_data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < m_clusters->size(); ++i) {
            double current_dist = euclidean_distance(m_clusters->at(i)->m_centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = i;
            }
        }
        if (m_clusters->at(best_cluster)->m_most_frequent_class == query_point->get_label()) {
            ++num_correct;
        }
    }
    double accuracy = 100.0 * (num_correct / (double)m_validation_data->size());
    std::cout << "validation completed with accuracy of " << accuracy << "%" << std::endl;
    return accuracy;
}

double KMeans::test() {
    std::cout << "Testing... ";
    double num_correct = 0.0;
    for (auto query_point: *m_test_data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < m_clusters->size(); ++i) {
            double current_dist = euclidean_distance(m_clusters->at(i)->m_centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = i;
            }
        }
        if (m_clusters->at(best_cluster)->m_most_frequent_class == query_point->get_label()) {
            ++num_correct;
        }
    }
    double accuracy = 100.0 * (num_correct / (double)m_test_data->size());
    std::cout << "Testing completed with accuracy of " << accuracy << "%" << std::endl;
    return accuracy;}

int main() {
    DataHandler *dh = new DataHandler();
    dh->read_feature_vector("../dataset/train-images-idx3-ubyte");
    dh->read_feature_labels("../dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    double performance = 0.0;
    double best_performance = 0.0;
    int best_k = 1;
    for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.002; ++k) {
        std::cout << "Set K to " << k << std::endl;
        KMeans *kmeans = new KMeans(k);
        kmeans->set_training_data(dh->get_training_data());
        kmeans->set_validation_data(dh->get_validation_data());
        kmeans->set_test_data(dh->get_test_data());
        kmeans->init_clusters();
        kmeans->train();
        performance = kmeans->validate();
        std::cout << "Current validation performance @ " << k << ": " << performance << std::endl;
        if (performance > best_performance) {
            best_performance = performance;
            best_k = k;
            std::cout << "Found a better K, which is " << best_k << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Running the best K, which is " << best_k << ", on the test set..." << std::endl;
    KMeans *kmeans = new KMeans(best_k);
    kmeans->set_training_data(dh->get_training_data());
    kmeans->set_validation_data(dh->get_validation_data());
    kmeans->set_test_data(dh->get_test_data());
    kmeans->init_clusters();
    performance = kmeans->test();
    std::cout << "The test performance with K=" << best_k << " are " << performance << std::endl;
}

