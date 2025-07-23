#include "../include/data_handler.hpp"

// Unittest - ETL
int main() {
    DataHandler *dh = new DataHandler();
    dh->read_input_data("../dataset/train-images-idx3-ubyte");
    dh->read_label_data("../dataset/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    delete dh;
    return 0;
}