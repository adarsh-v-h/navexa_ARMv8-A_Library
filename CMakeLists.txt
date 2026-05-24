cmake_minimum_required(VERSION 3.20)
# ADDED 'ASM' to languages to support inline assembly parsing
project(armv8lib VERSION 0.1.0 LANGUAGES CXX ASM)

# Bumped to C++20 (optional, but highly recommended for modern libraries)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Architecture flags (CRITICAL) ───────────────────────────────────────────
# +lse is required for atomic operations (CAS/LDADD)
# +memtag is required for MTE operations
add_compile_options(
    -march=armv8.5-a+memtag+lse
    -Wall
    -Wextra
)

# Collect all source files
file(GLOB_RECURSE LIB_SOURCES src/**/*.cpp)

# Build as a static library
add_library(armv8lib STATIC ${LIB_SOURCES})
target_include_directories(armv8lib PUBLIC include)

# Tests
enable_testing()
add_subdirectory(tests)