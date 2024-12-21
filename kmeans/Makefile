CC=g++
SRC := $(MNIST_ML_ROOT)/kmeans/src
CFLAGS := -std=c++11 -DEUCLID
INCLUDE_DIR := $(MNIST_ML_ROOT)/kmeans/include

all: main

main: $(SRC)/kmeans.cc
	$(CC) $(CFLAGS) $(SRC)/kmeans.cc -o main -L$(MNIST_ML_ROOT)/lib -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm main