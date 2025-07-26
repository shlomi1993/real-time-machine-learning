# Real-Time Machine Learning

A modular C++ implementation of classical machine learning algorithms designed for real-time inference and educational clarity. The project includes implementations of K-Nearest Neighbors (KNN), K-Means clustering, and feedforward Artificial Neural Networks (ANN), with support for the MNIST dataset, and partially for IRIS dataset.

## Project Structure

```
.
├── common         # Core data structures and preprocessing tools
├── dataset        # MNIST and IRIS datasets
├── models
│   ├── ann        # Feedforward Neural Network
│   ├── kmeans     # K-Means clustering algorithm
│   └── knn        # K-Nearest Neighbors classifier

````

Each model includes:
- `include/` header files
- `src/` source files
- `test.cpp` for basic usage
- `Makefile` for build automation

## Core Components

### `DataPoint`

Encapsulates a single data sample (an image). Supports:
- Raw and normalized features
- One-hot encoded labels
- Distance computation - for KNN/KMeans

### `DataHandler`

Handles all data-related operations:
- Reading binary or CSV files
- Normalizing features
- Counting classes
- Splitting data into train/validation/test sets

### `DataSet`

An abstract base class providing training, validation, and test datasets to models in a unified way.

## Models

### K-Nearest Neighbors (KNN)

A non-parametric classifier:
- Computes Euclidean distance between the query and training points
- Predicts label based on majority vote among the k-nearest neighbors
- Includes performance evaluation on validation/test sets

Source: `models/knn/`

### K-Means Clustering

Unsupervised clustering algorithm:
- Random or class-based centroid initialization
- Assigns points and updates centroids iteratively
- Assigns class by majority class within each cluster

Source: `models/kmeans/`

### Artificial Neural Network (ANN)
Simple feedforward neural network with:
- Multiple layers of `Neuron`s and `Layer`s
- Forward propagation, backpropagation, and weight updates
- Sigmoid activation function and its derivative

>  ⚠️ **Disclaimer:** On the Iris dataset, the neural network currently does not achieve competitive performance. Work is in progress to improve initialization, learning rate scheduling, and architecture tuning.

Source: `models/ann/`

## Dataset

- **MNIST**: Binary place files like `train-images-idx3-ubyte` in `dataset/`
- **Iris**: CSV file included as `iris.data`

Custom datasets can be integrated using `DataHandler::read_csv()` or `read_input_data()` and `read_label_data()` for binary formats.

## Building and Running

Each model directory contains a dedicated `Makefile`.

### Build and run KNN or KMeans

```bash
cd models/<knn/kmeans>
make
./bin/test.out
````

### Build and run ANN

```bash
cd models/amm
make <mnist/iris>
./bin/test.out
````
