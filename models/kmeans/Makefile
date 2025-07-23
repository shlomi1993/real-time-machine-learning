# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude -I../../common/include

# Directories
SRC_DIR := src
COMMON_DIR := ../../common
BIN_DIR := bin

# Files
SRCS := $(SRC_DIR)/kmeans.cpp $(SRC_DIR)/cluster.cpp \
        $(COMMON_DIR)/src/data_handler.cpp \
        $(COMMON_DIR)/src/data_point.cpp \
        $(COMMON_DIR)/src/data_set.cpp

OBJS := $(SRCS:.cpp=.o)
TARGET := $(BIN_DIR)/test.out

# Default target
all: $(TARGET)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Link object files into the final binary
$(TARGET): $(OBJS) test.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile the test file
test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile all other source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and binary
clean:
	rm -f *.o $(SRC_DIR)/*.o $(COMMON_DIR)/src/*.o $(TARGET)

# Run the built executable
run: all
	./$(TARGET)