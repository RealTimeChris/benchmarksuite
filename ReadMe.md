# Benchmark Suite

A modern, header-only C++20 microbenchmarking library for **CPU and CUDA**, built around a single goal: produce measurements you can defend to a skeptic.

Most benchmark numbers fall apart under scrutiny — too few iterations, the optimizer quietly deleting the work being timed, noise reported as a win, or results that can't be reproduced. Benchmark Suite is designed to close each of those gaps.

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

## Why These Measurements Are Trustworthy

Benchmarking is easy to get wrong in ways that favor whatever you're promoting. This library is built to remove the usual escape hatches:

- **Adaptive iteration sampling.** Iterations scale until throughput deviation falls below a configurable threshold, using sliding-window RSE analysis to find a stable measurement block. Stable code finishes fast; noisy code runs longer until it settles.
- **Dual convergence detection.** Measurements must satisfy both a Relative Standard Error (RSE) threshold *and* a mean-convergence criterion before results are accepted — reducing false convergence on temporarily stable but still-drifting measurements.
- **Welch's t-test for statistical ties.** Two statistically indistinguishable implementations are reported as a tie, not a win. Small, meaningless deltas don't get dressed up as victories.
- **Hardware performance counters.** Cross-platform cycles, instructions, IPC, cache behavior, cycles/byte, and instructions/byte — so a throughput claim can be cross-checked against the instruction-count delta that supposedly produced it.
- **Optimizer-resistant measurement.** `do_not_optimize_away()` prevents dead-code elimination from invalidating results — the classic way microbenchmarks silently measure nothing.
- **Reproducible by design.** vcpkg-installable, clone-and-run, with Markdown and CSV export for CI.
- **Non-invasive by design.** The library no longer propagates its own compiler flags or link options to consumers — your project keeps its own optimization settings.

It supports CPU and CUDA workloads through one API, including mixed CPU-vs-GPU comparisons, and runs on Windows, Linux, and macOS across MSVC, GCC, and Clang.

---

## Requirements

**Minimum:**
- A **C++20**-compliant compiler
- **GCC 13+** | **Clang 16+** | **MSVC 2022+**
- **CUDA 11.0+** (for GPU benchmarking)

CUDA support is not available on Apple Silicon.

---

## Installation

### Method 1: vcpkg + CMake (Recommended)

Add to your `vcpkg.json`:

```json
{
  "name": "your-project-name",
  "version": "1.0.0",
  "dependencies": [
    "rtc-benchmarksuite"
  ]
}
```

Wire it up in `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(YourProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmarksuite CONFIG REQUIRED)

add_executable(your_benchmark main.cpp)
target_link_libraries(your_benchmark PRIVATE benchmarksuite::benchmarksuite)
```

Configure with the vcpkg toolchain:

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### Method 2: Manual (Header-Only)

```bash
git clone https://github.com/RealTimeChris/benchmarksuite.git
```

```cmake
add_subdirectory(path/to/benchmarksuite)
target_include_directories(your_target PRIVATE path/to/benchmarksuite/include)
```

Then include the umbrella header:

```cpp
#include <bnch_swt>
```

---

## Basic Example

Comparing two integer-to-string conversion functions:

```cpp
#include <bnch_swt>

struct jsonifier_to_chars_benchmark {
    BNCH_SWT_HOST static uint64_t impl(std::vector<int64_t>& test_values,
                                       std::vector<std::string>& test_values_00,
                                       std::vector<std::string>& test_values_01) {
        uint64_t bytes_processed = 0;
        char newer_string[30]{};
        for (uint64_t x = 0; x < test_values.size(); ++x) {
            std::memset(newer_string, '\0', sizeof(newer_string));
            auto new_ptr = jsonifier_internal::to_chars(newer_string, test_values[x]);
            bytes_processed += test_values_00[x].size();
            test_values_01[x] = std::string{newer_string, static_cast<uint64_t>(new_ptr - newer_string)};
        }
        return bytes_processed;
    }
};

int main() {
    constexpr bnch_swt::stage_config_data config{
        .max_iteration_count = 1000,
        .measured_iteration_count = 25,
        .benchmark_type = bnch_swt::benchmark_types::cpu,
        .rse_threshold = 1.0,
        .convergence_threshold = 1.0,
        .max_time_in_s = 6
    };

    using benchmark = bnch_swt::benchmark_stage<"int-to-string-comparison", config>;

    benchmark::run_benchmark<"conversion-test", "jsonifier::to_chars", jsonifier_to_chars_benchmark>(
        test_values, test_values_00, test_values_01);

    auto test_results = benchmark::get_test_results("conversion-test");
    test_results.print();
    return 0;
}
```

A benchmark is a struct with a static `impl()` returning `uint64_t` (bytes processed). You group implementations under a test name, run each, then pull the ranked results.

---

## Configuration

The `stage_config_data` struct controls behavior:

```cpp
struct stage_config_data {
    bool clear_cpu_caches_before_iterations{ true };
    uint64_t measured_iteration_count{ 100 };
    uint64_t max_iteration_count{ 1000 };
    double convergence_threshold{ 1.0 };
    benchmark_types benchmark_type{};
    uint64_t max_time_in_s{ 5 };
    double rse_threshold{ 2.5 };
    uint64_t max_k{ 100000 };
    uint64_t min_k{ 10 };
};
```

| Field | Meaning |
| --- | --- |
| `clear_cpu_caches_before_iterations` | Evict CPU caches once before the run (default: true) |
| `measured_iteration_count` | Initial/minimum iterations per convergence round (default: 100) |
| `max_iteration_count` | Absolute upper bound on total iterations (default: 1000) |
| `convergence_threshold` | Max % change in the mean between rounds for mean-convergence (default: 1.0%) |
| `benchmark_type` | `benchmark_types::cpu` or `benchmark_types::cuda` |
| `max_time_in_s` | Hard time limit in whole seconds (default: 5) |
| `rse_threshold` | Max RSE (%) for RSE-convergence (default: 2.5%) |
| `max_k` / `min_k` | RSE sliding-window ceiling / floor (`min_k` must be > 1) |

Both `rse_threshold` and `convergence_threshold` must be satisfied simultaneously before a result is accepted as fully converged. If a time or iteration limit is hit first, the best result so far is returned with `converged = false`.

---

## Adaptive Benchmarking

1. Start with `measured_iteration_count` iterations.
2. Compute RSE over the last `k` samples (`k` bounded by `[min_k, max_k]`, scaling with total iterations).
3. Continue until both RSE ≤ `rse_threshold` **and** the per-round change in mean ≤ `convergence_threshold`.
4. Double the iteration count each round until convergence.
5. Stop early at `max_time_in_s` or `max_iteration_count`, returning the best result collected.

The `converged` field on each result records whether both criteria were met before the limits hit.

---

## Statistical Analysis

- **RSE-based convergence** rather than raw deviation range.
- **Mean convergence** as a second gate against false stability.
- **Welch's t-test tie detection** — proper unequal-variance handling when sample sizes differ. When `position_type_val` is `position_type::tie`, neither implementation is significantly faster under the available data.
- **Automated ranking** with proper tie grouping, plus win/loss/tie tracking aggregated across all tests via `stage_results_data`.
- **CSV and Markdown export** with a hardware-info preamble.

---

## CPU vs GPU Benchmarking

CPU and GPU implementations can be compared side by side through the `benchmark_type` field.

**CPU** benchmarks use `BNCH_SWT_HOST` and return `uint64_t` (bytes processed):

```cpp
struct cpu_computation_benchmark {
    BNCH_SWT_HOST static uint64_t impl(const std::vector<float>& input, std::vector<float>& output) {
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::sqrt(input[i] * input[i] + 1.0f);
        }
        return input.size() * sizeof(float);
    }
};
```

**CUDA** benchmarks use `BNCH_SWT_DEVICE`, return `void`, and contain kernel code:

```cpp
struct cuda_kernel_benchmark {
    BNCH_SWT_DEVICE static void impl(float* data, uint64_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = data[idx] * 2.0f;
        }
    }
};
```

GPU kernels are launched via `run_benchmark_from_host()`, with bytes-processed passed as a parameter. `run_benchmark_cooperative()` is available for kernels needing grid-wide synchronization.

---

## Working With Results

Results come from objects returned by `get_test_results()` and `get_all_results()`, giving you full control over format and destination.

`final_test_results` — per-test rankings:
- `.print(include_preamble = true)` — prints a Markdown table to stdout
- `.to_markdown(include_preamble, include_test_title, file_path = "")` — returns Markdown; optionally saves to disk
- `.to_csv(include_preamble = true, file_path = "")` — returns CSV; optionally saves to disk
- `.sorted_results` — `std::vector<library_completion_data>` sorted by throughput descending

`stage_results_data` — cross-test summary:
- `.to_csv(file_path = "")` — win/tie/loss summary across all tests
- `.results` — map of test name → `final_test_results`
- `.lib_positions` — `std::vector<library_positions>` sorted by win count

`library_completion_data` exposes per-implementation fields including `final_throughput` (MB/s), `final_rse`, `final_ms_spent`, `bytes_processed`, `final_sample_size`, `final_variance`, `final_mean`, `converged`, `position`, and `position_type_val`.

### Sample Markdown Output

```
### int-to-string-comparison Test Results
**CPU:** AMD Ryzen 9 7950X 16-Core Processor
**OS:** Linux-6.8.0
**Compiler:** GCC-13.2.0

| Library | Throughput (MB/s) | RSE (%) | Time (ms) | ... | Converged | Position |
| ------- | ----------------- | ------- | --------- | --- | --------- | -------- |
| jsonifier::to_chars | 84.58 | 1.23 | 5.79 | ... | true | 1 (Win) |
| glz::to_chars | 75.95 | 2.17 | 6.48 | ... | true | 2 (Loss) |
```

Statistically tied implementations are marked with `STATISTICAL TIE` in the Library column. When passed a `file_path`, output is saved to `file_path/<OS>-<Compiler>-<stage_name>.md` (or `.csv`).

---

## API Conventions

All APIs use `snake_case`:
- Functions: `run_benchmark()`, `get_test_results()`, `get_all_results()`
- Types: `stage_config_data`, `final_test_results`, `stage_results_data`, `library_completion_data`

---

## License

MIT © RealTimeChris. See [License.md](License.md).

For issues, feature requests, or contributions, visit the [GitHub repository](https://github.com/RealTimeChris/benchmarksuite).