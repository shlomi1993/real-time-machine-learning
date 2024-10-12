CC = g++
CFLAGS = -std=c++11 -g -fPIC
INCLUDE_DIR = $(MNIST_ML_ROOT)/include
SRC_DIR = $(MNIST_ML_ROOT)/src
OBJ_DIR = $(MNIST_ML_ROOT)/obj
LIB_DIR = $(MNIST_ML_ROOT)/lib
BIN_DIR = $(MNIST_ML_ROOT)/bin
LIB_DATA = libdata.so

# Automatically find all .cc source files and corresponding object files
SOURCES := $(wildcard $(SRC_DIR)/*.cc)
OBJECTS := $(patsubst $(SRC_DIR)/%.cc, $(OBJ_DIR)/%.o, $(SOURCES))

.PHONY: all clean libdir objdir bindir

all: $(LIB_DIR)/$(LIB_DATA) $(BIN_DIR)/main

# Rule to build the shared library
$(LIB_DIR)/$(LIB_DATA): libdir objdir $(OBJECTS)
	$(CC) $(CFLAGS) -shared -o $@ $(OBJECTS)

# Rule to build the main executable
$(BIN_DIR)/main: bindir $(MNIST_ML_ROOT)/main.cc $(LIB_DIR)/$(LIB_DATA)
	$(CC) $(CFLAGS) -o $@ $(MNIST_ML_ROOT)/main.cc -I$(INCLUDE_DIR) -L$(LIB_DIR) -ldata

# Generic rule for compiling object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Directory creation rules
libdir:
	mkdir -p $(LIB_DIR)

objdir:
	mkdir -p $(OBJ_DIR)

bindir:
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(LIB_DIR) $(OBJ_DIR) $(BIN_DIR)