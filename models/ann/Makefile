# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include

# Directories
SRC_DIR := src
COMMON_DIR := ../../common
BIN_DIR := bin

# Source files from this project and common
COMMON_SRCS := $(COMMON_DIR)/src/data_handler.cpp \
               $(COMMON_DIR)/src/data_set.cpp \
               $(COMMON_DIR)/src/data_point.cpp

SRCS := $(SRC_DIR)/layer.cpp \
        $(SRC_DIR)/neural_network.cpp \
        $(SRC_DIR)/neuron.cpp \
        $(COMMON_SRCS)

# Test source file
TEST_SRC := test.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)
TEST_OBJ := $(TEST_SRC:.cpp=.o)

# Target executable
TARGET := $(BIN_DIR)/test.out

# Default target (no dataset specified, error)
all:
	@echo "You must specify a dataset: make mnist OR make iris"

# Target for MNIST
mnist: CXXFLAGS += -DMNIST
mnist: $(TARGET)

# Target for IRIS
iris: CXXFLAGS += -DIRIS
iris: $(TARGET)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Link object files into the final binary
$(TARGET): $(OBJS) $(TEST_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile test file
$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and binary
clean:
	rm -f $(SRC_DIR)/*.o $(COMMON_DIR)/src/*.o *.o $(TARGET)

# Run the executable
run: all
	./$(TARGET)