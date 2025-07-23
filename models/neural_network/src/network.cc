
#include <numeric>
#include "data_handler.hpp"
#include "network.hpp"
#include "layer.hpp"

Network::Network(std::vector<int> spec, int input_size, int n_classes, double learning_rate) {
    for (int i = 0; i < spec.size(); ++i) {
        auto prev_layer_size = (i == 0) ? input_size : this->layers.at(i - 1)->neurons.size();
        this->layers.push_back(new Layer(prev_layer_size, spec.at(i)));
    }
    this->layers.push_back(new Layer(this->layers.at(this->layers.size() - 1)->neurons.size(), n_classes));
    this->learning_rate = learning_rate;
}

double Network::activate(std::vector<double> weights, std::vector<double> inputs) {
    double activation = weights.back(); // Bias term
    auto n_weights = weights.size() - 1;
    for (int i = 0; i < n_weights; ++i) {
        activation += weights[i] * inputs[i];
    }
    return activation;

}

double inline Network::transfer(double activation) {
    return 1.0 / (1.0 + exp(-activation));
}

double inline Network::transfer_derivative(double output) {
    return output * (1 - output);
}

std::vector<double> Network::fprop(Data *data) {
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < this->layers.size(); ++i) {
        Layer *layer = this->layers.at(i);
        std::vector<double> new_inputs;
        for (Neuron *n : layer->neurons) {
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation);
            new_inputs.push_back(n->output);
        }
        inputs = new_inputs;
    }
    return inputs;
}

void Network::bprop(Data *data) {
    auto n_non_input_layers = this->layers.size() - 1;
    for (int i = n_non_input_layers; i >= 0; --i) {
        Layer *layer = this->layers.at(i);
        std::vector<double> errors;
        if (i != n_non_input_layers) {
            for (int j = 0; j < layer->neurons.size(); ++j) {
                double error = 0.0;
                for (Neuron *n : this->layers.at(i + 1)->neurons) {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        } else {
            for (int j = 0; j < layer->neurons.size(); ++j) {
                Neuron *n = layer->neurons.at(j);
                double error = (double) data->get_class_vector().at(j) - n->output; // Calculate expected - actual
                errors.push_back(error);
            }
        }
        for (int j = 0; layer->neurons.size(); ++j) {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transfer_derivative(n->output); // Gradient / derivative part of back prop
        }
    }
}

void Network::update_weights(Data *data) {
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < this->layers.size(); ++i) {
        if (i != 0) {
            for (Neuron *n : this->layers.at(i - 1)->neurons) {
                inputs.push_back(n->output);
            }
        }
        for (Neuron *n : this->layers.at(i)->neurons) {
            for (int j = 0; j < this->layers.size(); ++j) {
                n->weights.at(j) += this->learning_rate * n->delta * inputs.at(j);
            }
            n->weights.back() += this->learning_rate * n->delta;
        }
        inputs.clear();
    }
}

int Network::predict(Data *data) {
    std::vector<double> outputs = fprop(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int n_epochs) {
    for (int i = 0;  i < n_epochs; ++i) {
        double sum_error = 0.0;
        for (Data *d : *this->training_data) {
            std::vector<double> outputs = this->fprop(d);
            std::vector<int> expected = d->get_class_vector();
            double temp_error_sum = 0.0;
            for (int j = 0; j < outputs.size(); ++j) {
                temp_error_sum += pow((double) expected.at(j) - outputs.at(j), 2);
            }
            sum_error += temp_error_sum;
            this->bprop(d);
            this->update_weights(d);
        }
        std::cout << "Epoch: " << i << "\tError: " << sum_error << std::endl;
    }
}

double Network::validate() {
    double n_correct = 0.0;
    double count = 0.0;
    for (Data *d : *this->validation_data) {
        ++count;
        int index = this->predict(d);
        if (d->get_class_vector().at(index) == 1) {
            ++n_correct;
        }
    }
    double accuracy = n_correct / count;
    return accuracy;
}

double Network::test() {
    double n_correct = 0.0;
    double count = 0.0;
    for (Data *d : *this->test_data) {
        ++count;
        int index = this->predict(d);
        if (d->get_class_vector().at(index) == 1) {
            ++n_correct;
        }
    }
    double accuracy = n_correct / count;
    return accuracy;
}

// Unittest - Neural-Network
int main() {
    DataHandler *data_handler = new DataHandler();
    data_handler->read_csv("../dataset/iris.data");
    data_handler->split_data();
    std::vector<int> hidden_layers = { 10 };

    int input_size = data_handler->get_training_data()->at(0)->get_normalized_feature_vector()->size();
    int n_classes = data_handler->get_class_counts();
    double learning_rate = 0.25;

    Network *net = new Network(hidden_layers, input_size, n_classes, learning_rate);
    net->set_training_data(data_handler->get_training_data());
    net->set_test_data(data_handler->get_test_data());
    net->set_validation_data(data_handler->get_validation_data());
    net->train(15);
    std::cout << "YOOO" << std::endl;
    net->validate();
    printf("Test Performance: %.3f\n", net->test());
}