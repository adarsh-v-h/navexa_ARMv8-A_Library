include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# Exclude mte test until it's fixed
file(GLOB_RECURSE TEST_SOURCES unit/*.cpp)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*test_mte.*")

add_executable(run_tests ${TEST_SOURCES})

target_link_libraries(run_tests
    PRIVATE
    armv8lib
    gtest
    gtest_main
)

target_include_directories(run_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(run_tests PROPERTIES
    CROSSCOMPILING_EMULATOR "qemu-aarch64-static;-L;/usr/aarch64-linux-gnu"
)

include(GoogleTest)
gtest_discover_tests(run_tests DISCOVERY_MODE PRE_TEST)
