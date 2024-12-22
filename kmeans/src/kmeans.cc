#include "kmeans.hpp"

KMeans::KMeans(int k) {
    this->n_clusters = k;
    this->clusters = new std::vector<Cluster *>();
    this->used_indexes = new std::unordered_set<int>();
}

void KMeans::init_clusters() {
    for (int i = 0; i < this->n_clusters; ++i) {
        int index = rand() % this->training_data->size();
        while (this->used_indexes->find(index) != this->used_indexes->end()) {
            index = rand() % this->training_data->size();
        }
        this->clusters->push_back(new Cluster(this->training_data->at(index)));
        this->used_indexes->insert(index);
    }
}

void KMeans::init_clusters_for_each_class() {
    std::unordered_set<int> classes_used;
    for (int i = 0; i < this->training_data->size(); ++i) {
        if (classes_used.find(this->training_data->at(i)->get_label()) == classes_used.end()) {
            this->clusters->push_back(new Cluster(this->training_data->at(i)));
            classes_used.insert(this->training_data->at(i)->get_label());
            this->used_indexes->insert(i);
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
    int index;
    do {
        index = rand() % this->training_data->size();
    } while (this->used_indexes->find(index) != this->used_indexes->end());
    double min_dist = std::numeric_limits<double>::max();
    int best_cluster = 0;
    for (int i = 0; i < this->clusters->size(); ++i) {
        double current_dist = euclidean_distance(this->clusters->at(i)->centroid, this->training_data->at(index));
        if (current_dist < min_dist) {
            min_dist = current_dist;
        }
    }
    this->clusters->at(best_cluster)->add_to_cluster(this->training_data->at(index));
    this->used_indexes->insert(index);
}

double KMeans::validate() {
    double num_correct = 0.0;
    for (auto query_point: *this->validation_data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < this->clusters->size(); ++i) {
            double current_dist = euclidean_distance(this->clusters->at(i)->centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = i;
            }
        }
        if (this->clusters->at(best_cluster)->most_frequent_class == query_point->get_label()) {
            ++num_correct;
        }
    }
    double accuracy = 100.0 * (num_correct / (double) this->validation_data->size());
    return accuracy;
}

double KMeans::test() {
    double num_correct = 0.0;
    for (auto query_point: *this->test_data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < this->clusters->size(); ++i) {
            double current_dist = euclidean_distance(this->clusters->at(i)->centroid, query_point);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = i;
            }
        }
        if (this->clusters->at(best_cluster)->most_frequent_class == query_point->get_label()) {
            ++num_correct;
        }
    }
    double accuracy = 100.0 * (num_correct / (double) this->test_data->size());
    return accuracy;
}

// Unittest - KMeans
int main() {
    DataHandler *dh = new DataHandler();
    dh->read_feature_vector("../dataset/train-images-idx3-ubyte");
    dh->read_feature_labels("../dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    double accuracy = 0.0;
    double best_accuracy = 0.0;
    int best_k = 1;
    for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.002; ++k) {
        KMeans *kmeans = new KMeans(k);
        kmeans->set_training_data(dh->get_training_data());
        kmeans->set_validation_data(dh->get_validation_data());
        kmeans->set_test_data(dh->get_test_data());
        kmeans->init_clusters();
        kmeans->train();
        accuracy = kmeans->validate();
        std::cout << "Validation accuracy with K=" << k << ": " << accuracy << "%";
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_k = k;
            std::cout << " (best so far)";
        }
        std::cout << std::endl;
    }

    std::cout << "Running the best K, which is " << best_k << ", on the test set..." << std::endl;
    KMeans *kmeans = new KMeans(best_k);
    kmeans->set_training_data(dh->get_training_data());
    kmeans->set_validation_data(dh->get_validation_data());
    kmeans->set_test_data(dh->get_test_data());
    kmeans->init_clusters();
    accuracy = kmeans->test();
    std::cout << "The test performance with K=" << best_k << " are " << accuracy << std::endl;
}

