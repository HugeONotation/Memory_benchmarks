#include <cstdint>
#include <climits>
#include <immintrin.h>
#include <benchmark/benchmark.h>

//=========================================================
// Configuration
//=========================================================

constexpr std::size_t l1_cache_size = 32 * 1024;
constexpr std::size_t repetitions = 16;

//=========================================================
// Constants
//=========================================================

constexpr std::size_t vector_size = sizeof(__m512i);
constexpr std::size_t iteration_count = (l1_cache_size / vector_size) - 1;
constexpr std::size_t page_size = 4 * 1024;

//=========================================================
// Data being read
//=========================================================

alignas(page_size) char test_buffer[l1_cache_size] = {};

//=========================================================
// Test benches
//=========================================================

template<unsigned Byte_offset>
void unalgined_test_bench(benchmark::State& state) {
    for (auto _ : state) {
        char* end = test_buffer + iteration_count * vector_size + Byte_offset;

        #pragma GCC unroll repetitions
        for (std::size_t i = 0; i < repetitions; ++i) {
            for (char* curr = test_buffer + Byte_offset; curr != end; curr += vector_size) {
                benchmark::DoNotOptimize(_mm512_loadu_si512(curr));
            }
        }
    }
}

void aligned_test_bench(benchmark::State& state) {
    const char* end = test_buffer + iteration_count * vector_size;
    for (auto _ : state) {
        #pragma GCC unroll repetitions
        for (std::size_t i = 0; i < repetitions; ++i) {
            for (char* curr = test_buffer; curr != end; curr += vector_size) {
                benchmark::DoNotOptimize(_mm512_load_si512(curr));
            }
        }
    }
}

//=========================================================
// Test bench registration
//=========================================================

auto unaligned_load_offset_0x00 = unalgined_test_bench<0x00>; BENCHMARK(unaligned_load_offset_0x00);
auto unaligned_load_offset_0x01 = unalgined_test_bench<0x01>; BENCHMARK(unaligned_load_offset_0x01);
auto unaligned_load_offset_0x02 = unalgined_test_bench<0x02>; BENCHMARK(unaligned_load_offset_0x02);
auto unaligned_load_offset_0x03 = unalgined_test_bench<0x03>; BENCHMARK(unaligned_load_offset_0x03);
auto unaligned_load_offset_0x04 = unalgined_test_bench<0x04>; BENCHMARK(unaligned_load_offset_0x04);
auto unaligned_load_offset_0x05 = unalgined_test_bench<0x05>; BENCHMARK(unaligned_load_offset_0x05);
auto unaligned_load_offset_0x06 = unalgined_test_bench<0x06>; BENCHMARK(unaligned_load_offset_0x06);
auto unaligned_load_offset_0x07 = unalgined_test_bench<0x07>; BENCHMARK(unaligned_load_offset_0x07);
auto unaligned_load_offset_0x08 = unalgined_test_bench<0x08>; BENCHMARK(unaligned_load_offset_0x08);
auto unaligned_load_offset_0x09 = unalgined_test_bench<0x09>; BENCHMARK(unaligned_load_offset_0x09);
auto unaligned_load_offset_0x0a = unalgined_test_bench<0x0a>; BENCHMARK(unaligned_load_offset_0x0a);
auto unaligned_load_offset_0x0b = unalgined_test_bench<0x0b>; BENCHMARK(unaligned_load_offset_0x0b);
auto unaligned_load_offset_0x0c = unalgined_test_bench<0x0c>; BENCHMARK(unaligned_load_offset_0x0c);
auto unaligned_load_offset_0x0d = unalgined_test_bench<0x0d>; BENCHMARK(unaligned_load_offset_0x0d);
auto unaligned_load_offset_0x0e = unalgined_test_bench<0x0e>; BENCHMARK(unaligned_load_offset_0x0e);
auto unaligned_load_offset_0x0f = unalgined_test_bench<0x0f>; BENCHMARK(unaligned_load_offset_0x0f);

auto unaligned_load_offset_0x10 = unalgined_test_bench<0x10>; BENCHMARK(unaligned_load_offset_0x10);
auto unaligned_load_offset_0x11 = unalgined_test_bench<0x11>; BENCHMARK(unaligned_load_offset_0x11);
auto unaligned_load_offset_0x12 = unalgined_test_bench<0x12>; BENCHMARK(unaligned_load_offset_0x12);
auto unaligned_load_offset_0x13 = unalgined_test_bench<0x13>; BENCHMARK(unaligned_load_offset_0x13);
auto unaligned_load_offset_0x14 = unalgined_test_bench<0x14>; BENCHMARK(unaligned_load_offset_0x14);
auto unaligned_load_offset_0x15 = unalgined_test_bench<0x15>; BENCHMARK(unaligned_load_offset_0x15);
auto unaligned_load_offset_0x16 = unalgined_test_bench<0x16>; BENCHMARK(unaligned_load_offset_0x16);
auto unaligned_load_offset_0x17 = unalgined_test_bench<0x17>; BENCHMARK(unaligned_load_offset_0x17);
auto unaligned_load_offset_0x18 = unalgined_test_bench<0x18>; BENCHMARK(unaligned_load_offset_0x18);
auto unaligned_load_offset_0x19 = unalgined_test_bench<0x19>; BENCHMARK(unaligned_load_offset_0x19);
auto unaligned_load_offset_0x1a = unalgined_test_bench<0x1a>; BENCHMARK(unaligned_load_offset_0x1a);
auto unaligned_load_offset_0x1b = unalgined_test_bench<0x1b>; BENCHMARK(unaligned_load_offset_0x1b);
auto unaligned_load_offset_0x1c = unalgined_test_bench<0x1c>; BENCHMARK(unaligned_load_offset_0x1c);
auto unaligned_load_offset_0x1d = unalgined_test_bench<0x1d>; BENCHMARK(unaligned_load_offset_0x1d);
auto unaligned_load_offset_0x1e = unalgined_test_bench<0x1e>; BENCHMARK(unaligned_load_offset_0x1e);
auto unaligned_load_offset_0x1f = unalgined_test_bench<0x1f>; BENCHMARK(unaligned_load_offset_0x1f);


auto unaligned_load_offset_0x20 = unalgined_test_bench<0x20>; BENCHMARK(unaligned_load_offset_0x20);
auto unaligned_load_offset_0x21 = unalgined_test_bench<0x21>; BENCHMARK(unaligned_load_offset_0x21);
auto unaligned_load_offset_0x22 = unalgined_test_bench<0x22>; BENCHMARK(unaligned_load_offset_0x22);
auto unaligned_load_offset_0x23 = unalgined_test_bench<0x23>; BENCHMARK(unaligned_load_offset_0x23);
auto unaligned_load_offset_0x24 = unalgined_test_bench<0x24>; BENCHMARK(unaligned_load_offset_0x24);
auto unaligned_load_offset_0x25 = unalgined_test_bench<0x25>; BENCHMARK(unaligned_load_offset_0x25);
auto unaligned_load_offset_0x26 = unalgined_test_bench<0x26>; BENCHMARK(unaligned_load_offset_0x26);
auto unaligned_load_offset_0x27 = unalgined_test_bench<0x27>; BENCHMARK(unaligned_load_offset_0x27);
auto unaligned_load_offset_0x28 = unalgined_test_bench<0x28>; BENCHMARK(unaligned_load_offset_0x28);
auto unaligned_load_offset_0x29 = unalgined_test_bench<0x29>; BENCHMARK(unaligned_load_offset_0x29);
auto unaligned_load_offset_0x2a = unalgined_test_bench<0x2a>; BENCHMARK(unaligned_load_offset_0x2a);
auto unaligned_load_offset_0x2b = unalgined_test_bench<0x2b>; BENCHMARK(unaligned_load_offset_0x2b);
auto unaligned_load_offset_0x2c = unalgined_test_bench<0x2c>; BENCHMARK(unaligned_load_offset_0x2c);
auto unaligned_load_offset_0x2d = unalgined_test_bench<0x2d>; BENCHMARK(unaligned_load_offset_0x2d);
auto unaligned_load_offset_0x2e = unalgined_test_bench<0x2e>; BENCHMARK(unaligned_load_offset_0x2e);
auto unaligned_load_offset_0x2f = unalgined_test_bench<0x2f>; BENCHMARK(unaligned_load_offset_0x2f);

auto unaligned_load_offset_0x30 = unalgined_test_bench<0x30>; BENCHMARK(unaligned_load_offset_0x30);
auto unaligned_load_offset_0x31 = unalgined_test_bench<0x31>; BENCHMARK(unaligned_load_offset_0x31);
auto unaligned_load_offset_0x32 = unalgined_test_bench<0x32>; BENCHMARK(unaligned_load_offset_0x32);
auto unaligned_load_offset_0x33 = unalgined_test_bench<0x33>; BENCHMARK(unaligned_load_offset_0x33);
auto unaligned_load_offset_0x34 = unalgined_test_bench<0x34>; BENCHMARK(unaligned_load_offset_0x34);
auto unaligned_load_offset_0x35 = unalgined_test_bench<0x35>; BENCHMARK(unaligned_load_offset_0x35);
auto unaligned_load_offset_0x36 = unalgined_test_bench<0x36>; BENCHMARK(unaligned_load_offset_0x36);
auto unaligned_load_offset_0x37 = unalgined_test_bench<0x37>; BENCHMARK(unaligned_load_offset_0x37);
auto unaligned_load_offset_0x38 = unalgined_test_bench<0x38>; BENCHMARK(unaligned_load_offset_0x38);
auto unaligned_load_offset_0x39 = unalgined_test_bench<0x39>; BENCHMARK(unaligned_load_offset_0x39);
auto unaligned_load_offset_0x3a = unalgined_test_bench<0x3a>; BENCHMARK(unaligned_load_offset_0x3a);
auto unaligned_load_offset_0x3b = unalgined_test_bench<0x3b>; BENCHMARK(unaligned_load_offset_0x3b);
auto unaligned_load_offset_0x3c = unalgined_test_bench<0x3c>; BENCHMARK(unaligned_load_offset_0x3c);
auto unaligned_load_offset_0x3d = unalgined_test_bench<0x3d>; BENCHMARK(unaligned_load_offset_0x3d);
auto unaligned_load_offset_0x3e = unalgined_test_bench<0x3e>; BENCHMARK(unaligned_load_offset_0x3e);
auto unaligned_load_offset_0x3f = unalgined_test_bench<0x3f>; BENCHMARK(unaligned_load_offset_0x3f);

auto aligned_load = aligned_test_bench; BENCHMARK(aligned_load);

//=========================================================
// Good ol' Main
//=========================================================

BENCHMARK_MAIN();
