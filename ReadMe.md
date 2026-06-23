# BenchmarkSuite

A header-only C++ benchmarking library with cross-platform hardware performance counter integration, providing precise measurements of cycles, and throughput with minimal overhead. Also supports CUDA GPU benchmarking.

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

## Features

- **Header-only** — just include and go, no linking required
- **Hardware performance counters** via native OS APIs:
  - macOS: kperf/kpc private frameworks (Apple Silicon + Intel)
  - Linux: perf_event / rdtsc
  - Windows: rdtsc / __rdtsc intrinsic
- **CUDA GPU benchmarking** — cudaEvent timing, cooperative kernel launches, SM/clock introspection
- **Adaptive convergence loop** — runs until RSE (Relative Standard Error) and mean stability thresholds are met, or a time/iteration budget expires
- **Statistical tie detection** — Welch's t-test with Welch-Satterthwaite degrees of freedom to distinguish real winners from noise
- **Cache eviction** — nuclear-grade random-access cache clearing between iterations for cold-start measurements
- **Thread affinity + priority pinning** — pins to P-cores on Intel hybrid CPUs, raises to REALTIME/SCHED_FIFO/QOS_USER_INTERACTIVE
- **Compile-time CPU/GPU property injection** — bakes cache sizes, alignment, SM count, etc. into the binary as constexpr
- **Multi-format output** — Markdown tables and CSV, with system info preambles
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

int main() {
    static constexpr bnch_swt::stage_config_data config{
        .measured_iteration_count = 100,
        .max_iteration_count = 10000,
        .rse_threshold = 2.5,
        .max_time_in_s = 5,
        .benchmark_type = bnch_swt::benchmark_types::cpu,
    };

    using bench = bnch_swt::benchmark_stage<"my-stage", config>;

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

The lambda returns the number of bytes processed — throughput calculations use this.

## Stage Configuration

`stage_config_data` controls the adaptive benchmarking loop:

- `measured_iteration_count` — initial epoch size (default 100)
- `max_iteration_count` — hard ceiling on total iterations (default 1000)
- `min_k` / `max_k` — statistical window bounds (default 30 / 100000)
- `rse_threshold` — target Relative Standard Error % for convergence (default 2.5)
- `convergence_threshold` — mean-stability threshold between epochs (default 1.0)
- `max_time_in_s` — wall-clock budget per benchmark (default 5)
- `clear_cpu_caches_before_iterations` — nuclear cache eviction between runs (default true)
- `benchmark_type` — `cpu` or `cuda`

The loop doubles the epoch size each iteration until both RSE and mean convergence criteria are satisfied, or the budget runs out.

## Output Formats

Results can be emitted as Markdown or CSV, with optional file writing:

```cpp
auto stage_results = bench::get_all_results();
for (const auto& test : stage_results.results) {
    test.to_markdown(true, true, "./output_dir");
    test.to_csv(true, "./output_dir");
}
stage_results.to_csv("./output_dir");
```

Output includes throughput (MB/s), RSE %, window duration, sample size, variance, latency, cycles/byte (when available), and position (Win/Tie/Loss).

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