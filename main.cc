#include "data_handler.hpp"

// Unittest - ETL
int main() {
    DataHandler *dh = new DataHandler();
    dh->read_feature_vector("./dataset/train-images-idx3-ubyte");
    dh->read_feature_labels("./dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}