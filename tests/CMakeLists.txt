include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_sve unit/test_sve.cpp)
target_link_libraries(test_sve armv8lib GTest::gtest_main)
target_compile_options(test_sve PRIVATE -march=armv8-a)
add_test(NAME sve COMMAND qemu-aarch64 -L /usr/aarch64-linux-gnu $<TARGET_FILE:test_sve>)

add_executable(test_mathext unit/test_mathext.cpp)
target_link_libraries(test_mathext armv8lib GTest::gtest_main)
target_compile_options(test_mathext PRIVATE -march=armv8-a)
add_test(NAME mathext COMMAND qemu-aarch64 -L /usr/aarch64-linux-gnu $<TARGET_FILE:test_mathext>)