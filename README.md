# Real-Time Machine Learning in C++

A modular C++ implementation of classical machine learning algorithms designed for real-time inference and educational clarity. The project includes implementations of K-Nearest Neighbors (KNN), K-Means clustering, and feedforward Artificial Neural Networks (ANN), with support for datasets such as MNIST and Iris.

---

## Project Structure

```

.
├── common         # Core data structures and preprocessing tools
├── dataset        # MNIST and Iris datasets
├── models
│   ├── ann        # Feedforward Neural Network
│   ├── kmeans     # K-Means clustering algorithm
│   └── knn        # K-Nearest Neighbors classifier

````

Each model includes:
- `include/` headers
- `src/` implementation files
- `test.cpp` for basic usage
- `Makefile` for build automation

---

## Core Components

### DataPoint
Encapsulates a single data sample (e.g., an image or a row of features). Supports:
- Raw and normalized features
- One-hot encoded labels
- Distance computation (for clustering/KNN)

### DataHandler
Handles all data-related operations:
- Reading binary or CSV files
- Normalizing features
- Counting classes
- Splitting data into train/validation/test sets

### DataSet
An abstract base class providing training, validation, and test datasets to models in a unified way.

---

## Models

### K-Nearest Neighbors (KNN)
A non-parametric classifier:
- Computes Euclidean distance between the query and training points
- Predicts label based on majority vote among the k-nearest neighbors
- Includes performance evaluation on validation/test sets

Source: `models/knn/`

---

### K-Means Clustering
Unsupervised clustering algorithm:
- Random or class-based centroid initialization
- Assigns points and updates centroids iteratively
- Assigns class by majority class within each cluster

Source: `models/kmeans/`

---

### Artificial Neural Network (ANN)
Simple feedforward neural network with:
- Multiple layers of `Neuron`s and `Layer`s
- Forward propagation, backpropagation, and weight updates
- Sigmoid activation function and its derivative

> **Disclaimer:** On the Iris dataset, the neural network currently does not achieve competitive performance. Work is in progress to improve initialization, learning rate scheduling, and architecture tuning.

Source: `models/ann/`

---

## Datasets

The project supports the following datasets:

- **MNIST** (Binary): Place files like `train-images-idx3-ubyte` in `dataset/`
- **Iris** (CSV): Included as `iris.data`

Custom datasets can be integrated using `DataHandler::read_csv()` or `read_input_data()` and `read_label_data()` for binary formats.

---

## Building and Running

Each model directory contains a dedicated `Makefile`.

### Example: Build and run ANN

```bash
cd models/ann
make
./bin/test.out
````

Repeat similarly for `knn` and `kmeans`.

---

## Features

* C++11+ codebase (C++17 recommended)
* Modular and reusable components
* No external dependencies
* Works on macOS and Linux
* Simple Makefile-based builds
* Educational and extensible design
