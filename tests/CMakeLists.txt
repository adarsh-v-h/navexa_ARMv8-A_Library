include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES unit/*.cpp)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*test_mte.*")

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE armv8lib
    PRIVATE
   
    GTest::gtest_main
)
target_include_directories(run_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(run_tests PRIVATE -march=armv8-a)
set_source_files_properties(unit/test_atomics.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.5-a+lse")
set_target_properties(run_tests PROPERTIES
    CROSSCOMPILING_EMULATOR "qemu-aarch64-static;-L;/usr/aarch64-linux-gnu"
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(run_tests DISCOVERY_MODE PRE_TEST)