# BenchmarkSuite

A header-only C++20 benchmarking library with cross-platform hardware performance counter integration, providing precise measurements of cycles, latency, and throughput with minimal overhead. Also supports CUDA GPU benchmarking.

### Compiler Support
![MSVC](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=microsoft&logoColor=green&label=MSVC&labelColor=pewter&color=blue)
![GCC](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=linux&logoColor=green&label=GCC&labelColor=pewter&color=blue)
![CLANG](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=apple&logoColor=green&label=CLANG&labelColor=pewter&color=blue)
![NVCC](https://img.shields.io/badge/NVCC-Supported-blue?style=plastic&logo=nvidia&logoColor=green&labelColor=pewter)

### Operating System Support
![Windows](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=microsoft&logoColor=green&label=Windows&labelColor=pewter&color=blue)
![Linux](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=linux&logoColor=green&label=Linux&labelColor=pewter&color=blue)
![Mac](https://img.shields.io/github/actions/workflow/status/realtimechris/benchmarksuite/unit-tests.yml?style=plastic&logo=apple&logoColor=green&label=MacOS&labelColor=pewter&color=blue)

---

## Used In
- **[StringInt-Benchmarks](https://github.com/nihilai-collective/StringInt-Benchmarks)** — string↔int conversion benchmarks comparing void-numerics, jeaiii_itoa, fmt, and the standard library, across Linux/Clang, Linux/GCC, Windows/MSVC, and macOS/Clang & GCC
- **[Json-Performance](https://github.com/nihilai-collective/Json-Performance)** — JSON parsing/serialization benchmarks comparing Jsonifier, Glaze, and Simdjson.

## Features

- **Header-only** — just include and go, no linking required
- **Hardware performance counters** via native OS APIs:
  - macOS: kperf/kpc private frameworks (Apple Silicon + Intel)
  - Linux: perf_event / rdtsc
  - Windows: rdtsc / __rdtsc intrinsic
- **CUDA GPU benchmarking** — cudaEvent timing, cooperative kernel launches, SM/clock introspection
- **Adaptive convergence loop** — doubles the epoch size each pass until both RSE and mean-stability thresholds are satisfied, or a time/iteration budget expires
- **Statistical tie detection** — Welch's t-test with Welch-Satterthwaite degrees of freedom to distinguish real winners from noise, with rank sharing across tied libraries
- **Cache eviction** — cache clearing between iterations for cold-start measurements, toggleable per stage
- **Thread affinity + priority pinning** — pins to P-cores on Intel hybrid CPUs, raises to REALTIME/SCHED_FIFO/QOS_USER_INTERACTIVE
- **Compile-time CPU/GPU property injection** — bakes cache sizes, alignment, SM count, etc. into the binary as constexpr
- **Multi-format output** — Markdown tables and CSV, with system info preambles, both per-test and per-stage (win/tie/loss rollups)
- **Do-not-optimize barriers** — compiler-specific inline asm to defeat DCE
- **Random data generation** — xoshiro256++ with time-based or deterministic seeding

## Requirements

- C++20 or later
- CMake 3.x
- Supported platforms:
  - Windows x64 (MSVC, Clang, GCC)
  - Linux x64/ARM64 (GCC, Clang)
  - macOS x64/ARM64 (AppleClang, GCC via Homebrew)
- Optional: CUDA toolkit for GPU benchmarks

## Installation

### vcpkg

```
vcpkg install rtc-benchmarksuite
```

Then in your CMakeLists.txt:

```cmake
find_package(benchmarksuite CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE benchmarksuite::benchmarksuite)
```

### FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  benchmarksuite
  GIT_REPOSITORY https://github.com/realtimechris/benchmarksuite.git
  GIT_TAG main
)
FetchContent_MakeAvailable(benchmarksuite)
target_link_libraries(your_target PRIVATE benchmarksuite::benchmarksuite)
```

## Quick Start

```cpp
#include <bnch_swt>

static constexpr bnch_swt::stage_config_data config{
    .clear_cpu_caches_before_iterations = true,
    .measured_iteration_count = 100,
    .max_iteration_count = 10000,
    .convergence_threshold = 1.0,
    .benchmark_type = bnch_swt::benchmark_types::cpu,
    .max_time_in_s = 5,
    .rse_threshold = 2.5,
    .max_k = 100000,
    .min_k = 30,
};

using bench = bnch_swt::benchmark_stage<"my-stage", config>;

int main() {
    bench::run_benchmark<"sort-test", "std-sort", +[](std::vector<int>& v) -> uint64_t {
        std::sort(v.begin(), v.end());
        bnch_swt::do_not_optimize_away(v);
        return v.size() * sizeof(int);
    }>(my_vector);

    auto results = bench::get_all_results();
    std::cout << results.results[0].to_markdown();
    return 0;
}
```

The functor/lambda returns the number of bytes processed — throughput calculations use this.

## Two Ways to Register a Benchmark

`run_benchmark` has two overloads, and they work fundamentally differently:

**1. Stateless lambda as a non-type template parameter** (shown above) — the `+[](...)` syntax decays the lambda to a function pointer, passed as `auto function` in the template argument list. This bakes the callable into the type itself at compile time. Requires no captures.

```cpp
bench::run_benchmark<"test-name", "library-name", +[](args...) -> uint64_t {
    return bytes_processed;
}>(runtime_args...);
```

**2. A functor type passed as a normal template argument**, invoked at runtime via a static `impl(metrics, args...)` method:

```cpp
struct my_functor {
    static uint64_t impl(bnch_swt::iteration_metrics& metrics, std::vector<int>& v) {
        std::sort(v.begin(), v.end());
        return v.size() * sizeof(int);
    }
};

bench::run_benchmark<"test-name", "library-name", my_functor>(my_vector);
```

Use the NTTP lambda form for quick one-off inline benchmarks. Use the functor form when the benchmark body needs to be reused, is non-trivial, or needs direct access to `iteration_metrics`.

## Example: Benchmarking Atomic Wait/Notify

Multi-threaded benchmarks work the same way — spin up your threads inside the functor's `impl`, join before returning. This example compares `std::atomic<uint64_t>` against `std::atomic_unsigned_lock_free` for wait/notify throughput, and shows `bnch_swt::pin_for_benchmark()` being called once up front to lock thread affinity/priority for the whole run.

```cpp
using namespace bnch_swt;

static constexpr uint64_t wait_notify_cycles{ 1000 };

struct test_atomic_uint64 {
    BNCH_SWT_HOST static uint64_t impl() {
        std::atomic<uint64_t> flag{ 0 };
        std::thread waiter([&]() {
            uint64_t value{};
            for (uint64_t i = 0; i < wait_notify_cycles; ++i) {
                uint64_t expected = i;
                ++value;
                flag.wait(expected);
                bnch_swt::do_not_optimize_away(value);
            }
        });
        uint64_t value{};
        for (uint64_t i = 1; i <= wait_notify_cycles; ++i) {
            flag.store(i, std::memory_order_release);
            flag.notify_one();
            value = flag.load();
            bnch_swt::do_not_optimize_away(value);
        }
        waiter.join();
        return 20000;
    }
};

struct test_atomic_signed_lock_free {
    BNCH_SWT_HOST static uint64_t impl() {
        std::atomic_unsigned_lock_free flag{ 0 };
        std::thread waiter([&]() {
            typename std::atomic_unsigned_lock_free::value_type value{};
            for (typename std::atomic_unsigned_lock_free::value_type i = 0; i < wait_notify_cycles; ++i) {
                typename std::atomic_unsigned_lock_free::value_type expected = i;
                ++value;
                flag.wait(expected);
                bnch_swt::do_not_optimize_away(value);
            }
        });
        typename std::atomic_unsigned_lock_free::value_type value{};
        for (typename std::atomic_unsigned_lock_free::value_type i = 1; i <= wait_notify_cycles; ++i) {
            flag.store(i, std::memory_order_release);
            flag.notify_one();
            value = flag.load();
            bnch_swt::do_not_optimize_away(value);
        }
        waiter.join();
        return 20000;
    }
};

int main() {
    using stage_type = benchmark_stage<"test_stage_01", stage_config_data{}>;
    bnch_swt::pin_for_benchmark();

    stage_type::run_benchmark<"test-test", "test_atomic_signed_lock_free", test_atomic_signed_lock_free>();
    stage_type::run_benchmark<"test-test", "test_atomic_uint64", test_atomic_uint64::impl>();

    auto test_rankings = stage_type::get_test_results("test-test");
    std::cout << test_rankings.to_csv() << std::endl;

    auto all_rankings = stage_type::get_all_results();
    std::cout << all_rankings.to_csv() << std::endl;
    return 0;
}
```

Note the two registration styles side by side: `test_atomic_signed_lock_free` is passed as a functor type (invoked via its `impl` method at runtime), while `test_atomic_uint64::impl` is passed directly as a function pointer NTTP. Both are valid — pick whichever reads cleaner for the call site.

## Stage Configuration

`stage_config_data` controls the adaptive benchmarking loop:

- `clear_cpu_caches_before_iterations` — cache eviction between runs (default `true`)
- `measured_iteration_count` — initial epoch size (default 100)
- `max_iteration_count` — hard ceiling on total iterations, also sizes the preallocated metrics buffer (default 1000)
- `convergence_threshold` — mean-stability threshold between epochs, as a percentage (default 1.0)
- `benchmark_type` — `cpu` or `cuda`
- `max_time_in_s` — wall-clock budget per benchmark (default 5)
- `rse_threshold` — target Relative Standard Error % for convergence (default 2.5)
- `max_k` — statistical window ceiling (default 100000)
- `min_k` — statistical window floor, must be > 1 for valid Bessel-corrected variance (default 30)

The loop doubles the epoch size each iteration until both RSE and mean convergence criteria are met, or the time/iteration budget runs out.

## Output Formats

Results can be emitted as Markdown or CSV, per-test or rolled up per-stage:

```cpp
auto stage_results = bench::get_all_results();
for (const auto& test : stage_results.results) {
    test.to_markdown(true, true, "./output_dir");
    test.to_csv(true, "./output_dir");
}
stage_results.to_csv("./output_dir");
```

Per-test output includes throughput (MB/s), RSE %, window duration, bytes processed, sample size, variance, latency, cycles/byte (when hardware counters are available), and Win/Tie/Loss position. Stage-level CSV rolls this up into win/tie/loss counts per library across all tests in the stage, alongside the stage config used to produce them.

## CUDA Support

Set `benchmark_type = bnch_swt::benchmark_types::cuda` and use the CUDA-specific launcher paths. See `unit-tests/main.cu` for a full example benchmarking native GPU division vs. Granlund-Montgomery magic-number division across constant memory, compile-time, and runtime dispatch paths.

## Statistical Methodology

- **Bessel's correction** on variance (dividing by k-1)
- **Welch's t-test** for pairwise comparison, tolerant of unequal variances and sample sizes
- **Welch-Satterthwaite** approximation for degrees of freedom
- **Rank sharing** — statistically tied libraries share the same position on the leaderboard

## Sanitizer Support

The unit-tests CMake exposes `BNCH_SWT_ASAN` and `BNCH_SWT_UBSAN` options. Note: UBSan has no MSVC equivalent, and GCC-on-macOS sanitizer combos are auto-disabled since they don't work.

## License

MIT © RealTimeChris — see License.md.

---