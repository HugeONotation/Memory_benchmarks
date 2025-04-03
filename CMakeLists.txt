cmake_minimum_required(VERSION 3.10)

project(Memory_tests)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
add_subdirectory(external/benchmark)

add_executable(memory_benchmarks
    ./src/Main.cpp
)

target_compile_options(memory_benchmarks PRIVATE "-march=native")

target_link_libraries(memory_benchmarks PRIVATE benchmark benchmark_main)
