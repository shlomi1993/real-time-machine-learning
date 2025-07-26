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

### Build and run Extract-Transform-Load (ETL)

```bash
cd common
make
./bin/test.out
````

#### ETL Output Example

```console
real-time-machine-learning % cd common 
common % make
mkdir -p obj
g++ -std=c++17 -Wall -Wextra -Iinclude -c src/data_handler.cpp -o obj/data_handler.o
g++ -std=c++17 -Wall -Wextra -Iinclude -c src/data_point.cpp -o obj/data_point.o
g++ -std=c++17 -Wall -Wextra -Iinclude -c src/data_set.cpp -o obj/data_set.o
mkdir -p bin
g++ -std=c++17 -Wall -Wextra -Iinclude test.cpp obj/data_handler.o obj/data_point.o obj/data_set.o -o bin/test.out
common % ./bin/test.out 
Input File Header read completed.
Successfully read and stored 60000 feature vectors.
Label File Header read completed.
Successfully read and stored labels.
Training data size: 45000.
Test data size: 12000.
Validation data size: 3000.
Successfully extracted 10 unique classes.

All tests passed successfully!
```

### Build and run KNN or KMeans

```bash
cd models/<knn/kmeans>
make
./bin/test.out
````

### KNN Output Example

```console
real-time-machine-learning % cd models/knn 
knn % make
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c src/knn.cpp -o src/knn.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_handler.cpp -o ../../common/src/data_handler.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_point.cpp -o ../../common/src/data_point.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_set.cpp -o ../../common/src/data_set.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c test.cpp -o test.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -o bin/test.out src/knn.o ../../common/src/data_handler.o ../../common/src/data_point.o ../../common/src/data_set.o test.o
knn % ./bin/test.out 
Input File Header read completed.
Successfully read and stored 60000 feature vectors.
Label File Header read completed.
Successfully read and stored labels.
Successfully extracted 10 unique classes.
Training data size: 45000.
Test data size: 12000.
Validation data size: 3000.
Set k to 2.
Current validation performance: 90.9091%
Current validation performance: 90.9091%
Current validation performance: 93.75%
Current validation performance: 93.0233%
Current validation performance: 92.5926%
Current validation performance: 93.75%
Current validation performance: 94.5946%
Current validation performance: 94.1176%
Current validation performance: 94.7368%
Current validation performance: 95.2381%
Current validation performance: 95.6522%
Current validation performance: 96%
...
```

#### KMeans Output Example

```console
real-time-machine-learning % cd models/kmeans 
kmeans % make
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c src/kmeans.cpp -o src/kmeans.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c src/cluster.cpp -o src/cluster.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_handler.cpp -o ../../common/src/data_handler.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_point.cpp -o ../../common/src/data_point.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c ../../common/src/data_set.cpp -o ../../common/src/data_set.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -c test.cpp -o test.o
mkdir -p bin
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -o bin/test.out src/kmeans.o src/cluster.o ../../common/src/data_handler.o ../../common/src/data_point.o ../../common/src/data_set.o test.o
kmeans % ./bin/test.out 
Input File Header read completed.
Successfully read and stored 60000 feature vectors.
Label File Header read completed.
Successfully read and stored labels.
Successfully extracted 10 unique classes.
Training data size: 45000.
Test data size: 12000.
Validation data size: 3000.
Current Performance @ K = 10: 57.4333
Current Performance @ K = 11: 58.9
Current Performance @ K = 12: 62.7333
Current Performance @ K = 13: 59.3333
Current Performance @ K = 14: 66
Current Performance @ K = 15: 67.2667
Current Performance @ K = 16: 64.1
Current Performance @ K = 17: 65.2333
Current Performance @ K = 18: 69.3333
Current Performance @ K = 19: 59.8333
Current Performance @ K = 20: 68.3333
...
```

### Build and run ANN

```bash
cd models/amm
make <mnist/iris>
./bin/test.out
````

#### ANN Output Example (MNIST)

```console
real-time-machine-learning % cd models/ann 
ann % make
You must specify a dataset: make mnist OR make iris
ann % make mnist
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -DMNIST -c src/layer.cpp -o src/layer.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -DMNIST -c src/neural_network.cpp -o src/neural_network.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -DMNIST -c src/neuron.cpp -o src/neuron.o
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -DMNIST -c test.cpp -o test.o
mkdir -p bin
clang++ -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include -DMNIST -o bin/test.out src/layer.o src/neural_network.o src/neuron.o ../../common/src/data_handler.o ../../common/src/data_set.o ../../common/src/data_point.o test.o
ann % ./bin/test.out 
Input File Header read completed.
Successfully read and stored 60000 feature vectors.
Label File Header read completed.
Successfully read and stored labels.
Successfully extracted 10 unique classes.
Training data size: 45000.
Test data size: 12000.
Validation data size: 3000.
Epoch: 0         Error = 19419.8314
Epoch: 1         Error = 7819.4686
Epoch: 2         Error = 6442.0395
Epoch: 3         Error = 5739.5664
Epoch: 4         Error = 5358.1860
Epoch: 5         Error = 5074.0687
Epoch: 6         Error = 4896.4492
Epoch: 7         Error = 4739.3658
Epoch: 8         Error = 4611.1628
Epoch: 9         Error = 4527.4296
Epoch: 10        Error = 4419.5625
Epoch: 11        Error = 4320.8683
Epoch: 12        Error = 4299.7197
Epoch: 13        Error = 4170.5805
Epoch: 14        Error = 4123.5465
Validation Performance: 0.9300
Test Performance: 0.926833
...
```
