# tests/CMakeLists.txt
find_package(GTest REQUIRED)
include(GoogleTest)

# Add unit tests
add_executable(test_atomics unit/test_atomics.cpp)
target_link_libraries(test_atomics PRIVATE armv8lib GTest::gtest_main)
gtest_discover_tests(test_atomics)

# Add integration tests
add_executable(test_combined integration/test_combined.cpp)
target_link_libraries(test_combined PRIVATE armv8lib GTest::gtest_main)
gtest_discover_tests(test_combined)
