cmake_minimum_required(VERSION 3.10)
project(kmeans)

set(CMAKE_CXX_STANDARD 11)

# Add the source directory
include_directories(${PROJECT_SOURCE_DIR})

# List your source files explicitly
set(SOURCES
    src/main.cpp
    src/kmeans.cpp
    src/cluster.cpp
    src/data_handler.cpp
)

add_executable(kmeans ${SOURCES})
