# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include

# Directories
SRC_DIR := src
COMMON_DIR := ../../common
BIN_DIR := bin

# Source files for this project and common
SRCS := $(SRC_DIR)/knn.cpp \
        $(COMMON_DIR)/src/data_handler.cpp \
        $(COMMON_DIR)/src/data_point.cpp \
        $(COMMON_DIR)/src/data_set.cpp

# Test source file
TEST_SRC := knn_test.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)
TEST_OBJ := $(TEST_SRC:.cpp=.o)

# Target executable
TARGET := $(BIN_DIR)/knn_test.out

# Default target
all: $(TARGET)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Link object files into the final binary
$(TARGET): $(OBJS) $(TEST_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile test file
$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile project and common source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and binary
clean:
	rm -f $(SRC_DIR)/*.o $(COMMON_DIR)/src/*.o *.o $(TARGET)

# Run the executable
run: all
	./$(TARGET)