cmake_minimum_required(VERSION 3.20)
project(armv8lib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all source files
file(GLOB_RECURSE LIB_SOURCES src/**/*.cpp)

# Build as a static library
add_library(armv8lib STATIC ${LIB_SOURCES})
target_include_directories(armv8lib PUBLIC include)

# Tests
enable_testing()
add_subdirectory(tests)