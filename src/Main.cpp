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

auto unaligned_load_offset_0x0 = unalgined_test_bench<0x0>;
auto unaligned_load_offset_0x1 = unalgined_test_bench<0x1>;
auto unaligned_load_offset_0x2 = unalgined_test_bench<0x2>;
auto unaligned_load_offset_0x3 = unalgined_test_bench<0x3>;
auto unaligned_load_offset_0x4 = unalgined_test_bench<0x4>;
auto unaligned_load_offset_0x5 = unalgined_test_bench<0x5>;
auto unaligned_load_offset_0x6 = unalgined_test_bench<0x6>;
auto unaligned_load_offset_0x7 = unalgined_test_bench<0x7>;
auto unaligned_load_offset_0x8 = unalgined_test_bench<0x8>;
auto unaligned_load_offset_0x9 = unalgined_test_bench<0x9>;
auto unaligned_load_offset_0xa = unalgined_test_bench<0xa>;
auto unaligned_load_offset_0xb = unalgined_test_bench<0xb>;
auto unaligned_load_offset_0xc = unalgined_test_bench<0xc>;
auto unaligned_load_offset_0xd = unalgined_test_bench<0xd>;
auto unaligned_load_offset_0xe = unalgined_test_bench<0xe>;
auto unaligned_load_offset_0xf = unalgined_test_bench<0xf>;

auto aligned_load = aligned_test_bench;

BENCHMARK(unaligned_load_offset_0x0);
BENCHMARK(unaligned_load_offset_0x1);
BENCHMARK(unaligned_load_offset_0x2);
BENCHMARK(unaligned_load_offset_0x3);
BENCHMARK(unaligned_load_offset_0x4);
BENCHMARK(unaligned_load_offset_0x5);
BENCHMARK(unaligned_load_offset_0x6);
BENCHMARK(unaligned_load_offset_0x7);
BENCHMARK(unaligned_load_offset_0x8);
BENCHMARK(unaligned_load_offset_0x9);
BENCHMARK(unaligned_load_offset_0xa);
BENCHMARK(unaligned_load_offset_0xb);
BENCHMARK(unaligned_load_offset_0xc);
BENCHMARK(unaligned_load_offset_0xd);
BENCHMARK(unaligned_load_offset_0xe);
BENCHMARK(unaligned_load_offset_0xf);

BENCHMARK(aligned_load);

//=========================================================
// Good ol' Main
//=========================================================

BENCHMARK_MAIN();
