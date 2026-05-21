# Benchmark Suite

Hello and welcome to Benchmark Suite! This is a modern, header-only C++20 benchmarking library with cross-platform hardware performance counter integration, providing precise measurements of cycles, instructions, branches, cache behavior, and throughput with minimal overhead.

The following operating systems and compilers are officially supported:

### Compiler Support
----
![MSVC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=MSVC&labelColor=pewter&color=blue&branch=main)
![GCC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=GCC&labelColor=pewter&color=blue&branch=main)
![CLANG](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=CLANG&labelColor=pewter&color=blue&branch=main)
![NVCC](https://img.shields.io/badge/NVCC-Supported-blue?style=plastic&logo=nvidia&logoColor=green&labelColor=pewter)

**Minimum Requirements:**
- **C++20** compliant compiler
- **GCC 13+** | **Clang 16+** | **MSVC 2022+**
- **CUDA 11.0+** (for GPU benchmarking)

### Operating System Support
----
![Windows](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=Windows&labelColor=pewter&color=blue&branch=main)
![Linux](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=Linux&labelColor=pewter&color=blue&branch=main)
![Mac](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=MacOS&labelColor=pewter&color=blue&branch=main)

---

# Quickstart Guide for benchmarksuite v1.0.2

This guide will walk you through setting up and running benchmarks using `benchmarksuite`.

## Table of Contents
- [Installation](#installation)
  - [Method 1: vcpkg + CMake (Recommended)](#method-1-vcpkg--cmake-recommended)
  - [Method 2: Manual Installation](#method-2-manual-installation)
  - [Requirements](#requirements)
- [Basic Example](#basic-example)
- [Creating Benchmarks](#creating-benchmarks)
- [Adaptive Benchmarking](#adaptive-benchmarking)
- [Statistical Analysis](#statistical-analysis)
- [CPU vs GPU Benchmarking](#cpu-vs-gpu-benchmarking)
- [Advanced Benchmark Methods](#advanced-benchmark-methods)
- [Running Benchmarks](#running-benchmarks)
  - [Common CMake Options](#common-cmake-options)
  - [Complete Project Example](#complete-project-example)
- [Output and Results](#output-and-results)
- [Features](#features)
- [API Conventions](#api-conventions)
- [Migrating from v1.0.0](#migrating-from-v100)

## Installation

### Method 1: vcpkg + CMake (Recommended)

**Step 1: Add to vcpkg.json**

Create or update your `vcpkg.json` in your project root:

```json
{
  "name": "your-project-name",
  "version": "1.0.0",
  "dependencies": [
    "rtc-benchmarksuite"
  ]
}
```

**Step 2: Configure CMake**

In your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(YourProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(benchmarksuite CONFIG REQUIRED)

add_executable(your_benchmark main.cpp)

target_link_libraries(your_benchmark PRIVATE benchmarksuite::benchmarksuite)

set_target_properties(your_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

**Step 3: Configure with vcpkg toolchain**

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

cmake --build build --config Release
```

**Step 4: Include in your code**

```cpp
#include <bnch_swt/index.hpp>

int main() {
    return 0;
}
```

### Method 2: Manual Installation

If not using vcpkg, you can include benchmarksuite as a header-only library:

**Step 1: Clone the repository**

```bash
git clone https://github.com/RealTimeChris/benchmarksuite.git
```

**Step 2: Add to CMake**

```cmake
add_subdirectory(path/to/benchmarksuite)

target_include_directories(your_target PRIVATE path/to/benchmarksuite/include)
```

**Step 3: Include headers**

```cpp
#include <bnch_swt/index.hpp>
```

### Requirements

To use `benchmarksuite`, ensure you have a **C++20 (or later)** compliant compiler.

**For CPU Benchmarking:**
- MSVC 2022 or later
- GCC 13 or later  
- Clang 16 or later

**For GPU/CUDA Benchmarking:**
- NVIDIA CUDA Toolkit 11.0 or later
- NVCC compiler
- CUDA-capable GPU

### Platform-Specific Notes

**Windows:**
- Use Visual Studio 2022 or later
- For CUDA: Install CUDA Toolkit from NVIDIA

**Linux:**
- Install build essentials: `sudo apt-get install build-essential`
- For CUDA: Install CUDA Toolkit via package manager or NVIDIA installer

**macOS:**
- Install Xcode Command Line Tools
- CUDA support not available on Apple Silicon (M1/M2/M3)

### Verification

Verify your installation with a simple test:

```cpp
#include <bnch_swt/index.hpp>
#include <iostream>

int main() {
    std::cout << "benchmarksuite successfully installed!" << std::endl;
    return 0;
}
```

## Basic Example
The following example demonstrates how to set up and run a benchmark comparing two integer-to-string conversion functions:

```cpp
struct glz_to_chars_benchmark {
    BNCH_SWT_HOST static uint64_t impl(std::vector<int64_t>& test_values, 
                                        std::vector<std::string>& test_values_00,
                                        std::vector<std::string>& test_values_01) {
        uint64_t bytes_processed = 0;
        char newer_string[30]{};
        for (uint64_t x = 0; x < test_values.size(); ++x) {
            std::memset(newer_string, '\0', sizeof(newer_string));
            auto new_ptr = glz::to_chars(newer_string, test_values[x]);
            bytes_processed += test_values_00[x].size();
            test_values_01[x] = std::string{newer_string, static_cast<uint64_t>(new_ptr - newer_string)};
        }
        return bytes_processed;
    }
};

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
    constexpr bnch_swt::stage_config config{
        .max_execution_count = 200,
        .measured_iteration_count = 25,
        .benchmark_type = bnch_swt::benchmark_types::cpu,
        .desired_percentage_deviation = 1.0,
        .max_time_seconds = 5.5
    };
    
    constexpr uint64_t count = 512;
    
    std::vector<int64_t> test_values = generate_random_integers<int64_t>(count, 20);
    std::vector<std::string> test_values_00;
    std::vector<std::string> test_values_01(count);
    
    for (uint64_t x = 0; x < count; ++x) {
        test_values_00.emplace_back(std::to_string(test_values[x]));
    }
    
    using benchmark = bnch_swt::benchmark_stage<"int-to-string-comparison", config>;
    
    benchmark::run_benchmark<"conversion-test", "glz::to_chars", glz_to_chars_benchmark>(
        test_values, test_values_00, test_values_01);
    benchmark::run_benchmark<"conversion-test", "jsonifier::to_chars", jsonifier_to_chars_benchmark>(
        test_values, test_values_00, test_values_01);
    
    benchmark::print_results();
    
    return 0;
}
```

## Creating Benchmarks
To create a benchmark:
1. Define your benchmark functions as structs with a static `impl()` method that returns `uint64_t` (bytes processed)
2. Use `bnch_swt::benchmark_stage` with `stage_config` for configuration
3. Call `run_benchmark` with test name, subject name, benchmark struct, and arguments

### Benchmark Stage
The `benchmark_stage` structure orchestrates each test and supports both CPU and GPU benchmarking:

```cpp
template<bnch_swt::string_literal stage_name,
         bnch_swt::stage_config stage_config_new = bnch_swt::stage_config{},
         bnch_swt::string_literal metric_name = bnch_swt::string_literal<1>{}
>
struct benchmark_stage;

// Default configuration
using cpu_benchmark = bnch_swt::benchmark_stage<"my-benchmark">;

// Custom configuration
constexpr bnch_swt::stage_config gpu_config{
    .max_execution_count = 100,
    .measured_iteration_count = 10,
    .benchmark_type = bnch_swt::benchmark_types::cuda,
    .clear_cpu_cache_between_each_iteration = false,
    .clear_cpu_cache_before_all_iterations = true,
    .desired_percentage_deviation = 0.5,
    .max_time_seconds = 3.0
};
using gpu_benchmark = bnch_swt::benchmark_stage<"gpu-test", gpu_config>;

// Custom metric name
using compression_bench = bnch_swt::benchmark_stage<"compression", stage_config{}, "compression-ratio">;
```

### Stage Configuration
The `stage_config` struct controls benchmark behavior:

```cpp
struct stage_config {
    uint64_t max_execution_count{ 200 };                    // Maximum iterations (including warmup)
    uint64_t measured_iteration_count{ 10 };                // Number of iterations to measure
    benchmark_types benchmark_type{ benchmark_types::cpu }; // CPU or CUDA
    bool clear_cpu_cache_between_each_iteration{ false };   // Clear cache between iterations
    bool clear_cpu_cache_before_all_iterations{ true };     // Clear cache before starting
    double desired_percentage_deviation{ 1.0 };              // Target stability threshold (%)
    double max_time_seconds{ 5.5 };                         // Maximum runtime limit
};
```

### Methods

#### `run_benchmark<test_name, subject_name, function_type>(args...)`
Executes the benchmark using a struct with a static `impl()` method. The benchmark automatically scales iterations until statistical stability is reached.

**Parameters:**
- **test_name**: String literal grouping related benchmarks together
- **subject_name**: String literal identifying this specific implementation
- **function_type**: Struct type with a static `impl()` method
- **args...**: Arguments forwarded to the `impl()` method

**Returns:** Reference to `performance_metrics<benchmark_type>` object

**Example:**
```cpp
struct my_benchmark {
    BNCH_SWT_HOST static uint64_t impl(std::vector<int>& data) {
        uint64_t sum = 0;
        for (auto& val : data) {
            sum += val;
        }
        return data.size() * sizeof(int);
    }
};

constexpr bnch_swt::stage_config config{ .max_execution_count = 500, .measured_iteration_count = 50 };
using bench = bnch_swt::benchmark_stage<"test", config>;
std::vector<int> data(1000);
bench::run_benchmark<"math-test", "my-implementation", my_benchmark>(data);
```

#### `run_benchmark<test_name, subject_name, function>(args...)`
Executes the benchmark using a function or lambda directly (passed as non-type template parameter).

**Parameters:**
- **test_name**: String literal grouping related benchmarks
- **subject_name**: String literal identifying this specific implementation
- **function**: Function or lambda to benchmark (as non-type template parameter)
- **args...**: Arguments forwarded to the function

**Example:**
```cpp
constexpr auto my_lambda = [](std::vector<int>& data) -> uint64_t {
    uint64_t sum = 0;
    for (auto& val : data) {
        sum += val;
    }
    return data.size() * sizeof(int);
};

constexpr bnch_swt::stage_config config{ .max_execution_count = 500, .measured_iteration_count = 50 };
using bench = bnch_swt::benchmark_stage<"test", config>;
std::vector<int> data(1000);
bench::run_benchmark<"math-test", "my-implementation", my_lambda>(data);
```

#### `run_benchmark_from_host<test_name, subject_name, function_type>(bytes_processed, args...)`
Executes CUDA benchmarks launched from host code.

**Parameters:**
- **test_name**: String literal grouping related benchmarks
- **subject_name**: String literal identifying this specific implementation
- **function_type**: Function type to benchmark
- **bytes_processed**: Number of bytes processed per iteration
- **args...**: Arguments forwarded to the function

**Example:**
```cpp
struct cuda_host_launcher {
    static void impl(float* gpu_data, uint64_t size) {
        dim3 grid{256};
        dim3 block{256};
        my_kernel<<<grid, block>>>(gpu_data, size);
        cudaDeviceSynchronize();
    }
};

constexpr bnch_swt::stage_config config{ 
    .benchmark_type = bnch_swt::benchmark_types::cuda,
    .measured_iteration_count = 10
};
using bench = bnch_swt::benchmark_stage<"cuda-test", config>;
float* gpu_data;
cudaMalloc(&gpu_data, 1024 * sizeof(float));
bench::run_benchmark_from_host<"kernel-test", "my-kernel", cuda_host_launcher>(
    1024 * sizeof(float), gpu_data, 1024);
```

#### `run_benchmark_cooperative<test_name, subject_name, function>(args...)`
Executes CUDA cooperative group kernels requiring grid-wide synchronization.

**Parameters:**
- **test_name**: String literal grouping related benchmarks
- **subject_name**: String literal identifying this specific implementation
- **function**: Function to benchmark (as non-type template parameter)
- **args...**: Arguments forwarded to the function

#### `print_results<metrics_presence>(show_metrics)`
Displays performance metrics with statistical analysis, rankings, and confidence intervals.

**Parameters:**
- **metrics_presence**: Template parameter controlling which metrics to display
- **show_metrics**: Whether to show detailed hardware counter metrics

**Example:**
```cpp
benchmark::print_results();  // Default metrics

// Custom metric selection
bnch_swt::performance_metrics_presence<bnch_swt::benchmark_types::cpu> custom_metrics{};
custom_metrics.throughput_mb_per_sec = true;
custom_metrics.cycles_per_byte = true;
custom_metrics.instructions_per_cycle = true;
benchmark::print_results<custom_metrics>(true);
```

#### `generate_markdown(title, file_path)`
Generates a formatted Markdown report of all benchmark results.

**Parameters:**
- **title**: Title for the report
- **file_path**: Optional directory path to save the report (auto-named with OS and compiler)

**Returns:** `std::string` containing the Markdown report

**Example:**
```cpp
auto report = benchmark::generate_markdown("Performance Analysis", "./results/");
std::cout << report << std::endl;
```

#### `get_all_results()`
Returns all results organized by test name.

**Returns:** `std::vector<stage_results<stage_name, benchmark_type>::test_results>`

#### `get_test_results(test_name)`
Returns results for a specific test name.

**Returns:** `std::unordered_map<std::string_view, performance_metrics<benchmark_type>>`

#### `clear_all_results()`
Resets all collected results for the stage.

### Benchmark Function Requirements
Benchmark functions must be defined as structs with a static `impl()` method:

**For CPU benchmarks:**
```cpp
struct my_cpu_benchmark {
    BNCH_SWT_HOST static uint64_t impl(/* your parameters */) {
        uint64_t bytes_processed = /* calculate bytes */;
        return bytes_processed;
    }
};
```

**For CUDA benchmarks:**
```cpp
struct my_cuda_benchmark {
    BNCH_SWT_DEVICE static void impl(/* your parameters */) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // kernel code here
    }
};
```

**Key differences:**
- **CPU**: `impl()` returns `uint64_t` (bytes processed) and uses `BNCH_SWT_HOST`
- **CUDA**: `impl()` returns `void`, uses `BNCH_SWT_DEVICE`, and contains kernel code
- **CUDA**: Bytes processed is passed as a parameter to `run_benchmark_from_host()`

## Adaptive Benchmarking

As of v1.0.2, benchmarksuite features **adaptive iteration scaling** that automatically determines the optimal number of iterations for statistical stability.

### How It Works

1. **Starts with small iteration count**: Begins with `measured_iteration_count * 2` iterations
2. **Sliding window analysis**: Evaluates all consecutive windows of `measured_iteration_count` iterations
3. **Stability detection**: Continues until throughput deviation ≤ `desired_percentage_deviation`
4. **Iteration doubling**: Doubles iteration count each round until stability or limits reached
5. **Time protection**: Automatically stops after `max_time_seconds` to prevent excessively long runs

### Configuration Example

```cpp
constexpr bnch_swt::stage_config precise_config{
    .max_execution_count = 10000,           // Upper limit
    .measured_iteration_count = 50,          // Window size for analysis
    .desired_percentage_deviation = 0.5,     // Target: 0.5% stability
    .max_time_seconds = 10.0                // Max 10 seconds per benchmark
};

using bench = bnch_swt::benchmark_stage<"precise-benchmark", precise_config>;
```

### Benefits

- **No more guessing**: No need to manually tune iteration counts
- **Comparable results**: All benchmarks achieve similar statistical confidence
- **Time-efficient**: Stops early for stable code, continues longer for noisy measurements
- **Reproducible**: Same configuration produces consistent stability across runs

## Statistical Analysis

Benchmark results now include **95% confidence interval analysis** with automatic tie detection and ranking.

### Statistical Features

- **Confidence intervals**: Calculated from throughput deviation percentages
- **Statistical tie detection**: Identifies when implementations are statistically indistinguishable
- **Automated ranking**: Orders results with proper tie handling
- **Win/loss/tie tracking**: Summary statistics across multiple tests
- **Markdown export**: Professional reports for documentation

### Output Example

```
=== STATISTICAL SUMMARY FOR int-to-string-comparison ===
(95% confidence intervals, statistical ties don't count as wins)

jsonifier::to_chars: 1 wins
glz::to_chars: 0 wins (1 second place)

=== STATISTICAL TIES (no clear winner) ===
fast_float: 2 tests where statistically tied for first
```

### Understanding Statistical Ties

When two implementations have overlapping confidence intervals, they are considered **statistically tied** - neither is significantly faster than the other. This is reported clearly to prevent over-interpretation of small performance differences.

## CPU vs GPU Benchmarking

Benchmarksuite supports both CPU and GPU (CUDA) benchmarking through the `benchmark_type` enum in `stage_config`.

### CPU Benchmarks
```cpp
struct cpu_computation_benchmark {
    BNCH_SWT_HOST static uint64_t impl(const std::vector<float>& input, std::vector<float>& output) {
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::sqrt(input[i] * input[i] + 1.0f);
        }
        return input.size() * sizeof(float);
    }
};

constexpr bnch_swt::stage_config cpu_config{
    .benchmark_type = bnch_swt::benchmark_types::cpu,
    .max_execution_count = 200,
    .measured_iteration_count = 25
};

using cpu_stage = bnch_swt::benchmark_stage<"cpu-test", cpu_config>;

constexpr size_t data_size = 1024 * 1024;
std::vector<float> input(data_size, 1.0f);
std::vector<float> output(data_size);

cpu_stage::run_benchmark<"math-test", "cpu-impl", cpu_computation_benchmark>(input, output);
cpu_stage::print_results();
```

### GPU/CUDA Benchmarks
```cpp
struct cuda_kernel_benchmark {
    BNCH_SWT_DEVICE static void impl(float* data, uint64_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = data[idx] * 2.0f;
        }
    }
};

constexpr bnch_swt::stage_config gpu_config{
    .benchmark_type = bnch_swt::benchmark_types::cuda,
    .max_execution_count = 100,
    .measured_iteration_count = 10,
    .max_time_seconds = 5.0
};

using cuda_stage = bnch_swt::benchmark_stage<"gpu-test", gpu_config>;

constexpr uint64_t data_size = 1024 * 1024;
float* gpu_data;
cudaMalloc(&gpu_data, data_size * sizeof(float));

dim3 grid{256, 1, 1};
dim3 block{256, 1, 1};
uint64_t bytes_processed = data_size * sizeof(float);

cuda_stage::run_benchmark_from_host<"kernel-test", "gpu-impl", cuda_kernel_benchmark>(
    bytes_processed, grid, block, 0, gpu_data, data_size);

cuda_stage::print_results();
cudaFree(gpu_data);
```

### Mixed CPU/GPU Benchmarking
Compare CPU and GPU implementations side-by-side:

```cpp
constexpr uint64_t data_size = 1024 * 1024;

// CPU implementation
struct cpu_process {
    BNCH_SWT_HOST static uint64_t impl(std::vector<float>& cpu_data) {
        for (size_t i = 0; i < cpu_data.size(); ++i) {
            cpu_data[i] = cpu_data[i] * 2.0f;
        }
        return cpu_data.size() * sizeof(float);
    }
};

// GPU implementation
struct gpu_process {
    BNCH_SWT_DEVICE static void impl(float* gpu_data, uint64_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            gpu_data[idx] = gpu_data[idx] * 2.0f;
        }
    }
};

// Run both
constexpr bnch_swt::stage_config config{ .max_execution_count = 100, .measured_iteration_count = 10 };
using stage = bnch_swt::benchmark_stage<"cpu-vs-gpu", config>;

std::vector<float> cpu_data(data_size);
float* gpu_data;
cudaMalloc(&gpu_data, data_size * sizeof(float));

stage::run_benchmark<"vector-multiply", "cpu-version", cpu_process>(cpu_data);

dim3 grid{(data_size + 255) / 256, 1, 1};
dim3 block{256, 1, 1};
stage::run_benchmark_from_host<"vector-multiply", "gpu-version", gpu_process>(
    data_size * sizeof(float), grid, block, 0, gpu_data, data_size);

stage::print_results();
cudaFree(gpu_data);
```

### Cache Clearing Option
For accurate cold-cache CPU benchmarks:

```cpp
constexpr bnch_swt::stage_config cold_cache_config{
    .clear_cpu_cache_between_each_iteration = true,
    .clear_cpu_cache_before_all_iterations = true,
    .max_execution_count = 200,
    .measured_iteration_count = 25
};

using cold_bench = bnch_swt::benchmark_stage<"cache-test", cold_cache_config>;
```

### Custom Metrics
Specify custom metric names for specialized benchmarks:

```cpp
constexpr bnch_swt::stage_config config{ .max_execution_count = 200, .measured_iteration_count = 25 };
using compression_bench = bnch_swt::benchmark_stage<"compression-test", config, "compression-ratio">;

struct compress_benchmark {
    BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& input) {
        auto compressed = compress_data(input);
        return (input.size() * 1000) / compressed.size();  // Ratio * 1000
    }
};

compression_bench::run_benchmark<"compression", "my-compressor", compress_benchmark>(input_data);
compression_bench::print_results();  // Shows "compression-ratio" instead of MB/s
```

## Advanced Benchmark Methods

### Host-Launched Kernels
Use `run_benchmark_from_host()` for custom CUDA kernel configurations:

```cpp
struct custom_launcher {
    static void impl(float* data, uint64_t size, int shared_bytes) {
        dim3 grid{static_cast<unsigned int>((size + 255) / 256)};
        dim3 block{256};
        my_kernel<<<grid, block, shared_bytes>>>(data, size);
        cudaDeviceSynchronize();
    }
};

constexpr bnch_swt::stage_config config{ .benchmark_type = bnch_swt::benchmark_types::cuda };
using bench = bnch_swt::benchmark_stage<"custom-kernel", config>;

bench::run_benchmark_from_host<"launch-test", "custom", custom_launcher>(
    data_size * sizeof(float), gpu_data, data_size, 4096);
```

### Cooperative Kernels
Use `run_benchmark_cooperative()` for kernels requiring grid-wide sync:

```cpp
constexpr auto cooperative_reduce = [](float* data, float* result, uint64_t size) -> uint64_t {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    // reduction logic here
    grid.sync();
    return size * sizeof(float);
};

bench::run_benchmark_cooperative<"reduce-test", "grid-reduce", cooperative_reduce>(
    grid, block, shared_mem, stream, bytes_processed, gpu_data, gpu_result, size);
```

## Running Benchmarks

**With vcpkg + CMake (recommended):**

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release

./build/your_benchmark
.\build\Release\your_benchmark.exe
```

**Manual CMake build:**

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/your_benchmark
```

**For CUDA benchmarks, specify target architecture:**

```bash
cmake -B build -S . \
  -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
  
cmake --build build --config Release
```

### Common CMake Options

- `-DCMAKE_BUILD_TYPE=Release` - Build optimized release version
- `-DCMAKE_CUDA_ARCHITECTURES=86` - Target specific CUDA compute capability (e.g., 86 for RTX 30xx/40xx)
- `-DCMAKE_CXX_COMPILER=clang++` - Specify C++ compiler
- `-DCMAKE_CUDA_COMPILER=nvcc` - Specify CUDA compiler

### Complete Project Example

**Project structure:**
```
my-benchmark/
├── CMakeLists.txt
├── vcpkg.json
├── main.cpp
└── benchmarks/
    ├── cpu_benchmark.hpp
    └── gpu_benchmark.cuh
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(MyBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(benchmarksuite CONFIG REQUIRED)

add_executable(my_benchmark 
    main.cpp
    benchmarks/cpu_benchmark.hpp
    benchmarks/gpu_benchmark.cuh
)

target_link_libraries(my_benchmark PRIVATE 
    benchmarksuite::benchmarksuite
)

set_target_properties(my_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(MSVC)
    target_compile_options(my_benchmark PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(my_benchmark PRIVATE -O3 -march=native)
endif()
```

**vcpkg.json:**
```json
{
  "name": "my-benchmark",
  "version": "1.0.0",
  "dependencies": [
    "rtc-benchmarksuite"
  ]
}
```

## Output and Results

### Standard Output
```
----------------------------------------
CPU Performance Metrics for Stage: int-to-string-comparison
Running on: AMD Ryzen 9 7950X 16-Core Processor
OS: Linux 6.8.0
Compiler: GNU 13.2.0
----------------------------------------
Test: conversion-test
----------------------------------------
1. jsonifier::to_chars (84.58 MB/s +/-1.23%) | ~11.36% faster than glz::to_chars
----------------------------------------
Metrics for: jsonifier::to_chars
Total Iterations to Stabilize: 394
Measured Iterations: 20
Bytes Processed: 512.00
Nanoseconds per Execution: 5785.25
Frequency (GHz): 4.83
Throughput (MB/s): 84.58
Throughput Percentage Deviation (+/-%): 1.23
Cycles per Execution: 27921.20
Cycles per Byte: 54.53
Instructions per Execution: 52026.00
Instructions per Cycle: 1.86
Instructions per Byte: 101.61
----------------------------------------
2. glz::to_chars (75.95 MB/s +/-2.17%)
----------------------------------------
Metrics for: glz::to_chars
Total Iterations to Stabilize: 421
Measured Iterations: 20
Bytes Processed: 512.00
Nanoseconds per Execution: 6480.30
Throughput (MB/s): 75.95
Throughput Percentage Deviation (+/-%): 2.17
Cycles per Execution: 30314.40
Cycles per Byte: 59.21
Instructions per Execution: 51513.00
Instructions per Cycle: 1.70
----------------------------------------

=== STATISTICAL SUMMARY FOR int-to-string-comparison ===
(95% confidence intervals, statistical ties don't count as wins)

jsonifier::to_chars: 1 wins
glz::to_chars: 0 wins (1 second place)
```

### Markdown Report Generation
```cpp
auto report = benchmark::generate_markdown("Performance Analysis", "./results/");
```

Generates formatted Markdown files with complete statistical analysis, perfect for CI/CD documentation.

## Features

### Dual Benchmarking Support
- **CPU Benchmarking**: Traditional CPU performance measurement with hardware counters
- **GPU/CUDA Benchmarking**: Native CUDA kernel benchmarking with grid/block configuration
- **Mixed Workloads**: Compare CPU vs GPU implementations side-by-side
- **Automatic Device Selection**: Choose benchmark type via `stage_config`

### Adaptive Benchmarking (v1.0.2+)
- **Automatic iteration scaling**: Dynamically increases iterations until statistical stability
- **Sliding window analysis**: Finds optimal consecutive block of iterations
- **Percentage deviation targeting**: Configurable stability threshold (default: 1%)
- **Time-based termination**: Maximum runtime protection (default: 5.5 seconds)

### Statistical Analysis (v1.0.2+)
- **Confidence intervals**: 95% confidence intervals for throughput comparisons
- **Statistical tie detection**: Automatically identifies indistinguishable implementations
- **Automated ranking**: Orders results with proper tie handling
- **Win/loss/tie tracking**: Summary statistics across multiple tests
- **Markdown export**: Professional reports for documentation

### Advanced Execution Modes
- **Standard Benchmarking**: Default `run_benchmark()` with adaptive iteration scaling
- **Host-Launched Kernels**: `run_benchmark_from_host()` for custom kernel launch configurations
- **Cooperative Groups**: `run_benchmark_cooperative()` for grid-wide synchronization
- **Function or Struct**: Support for both function-based and struct-based benchmarks

### Advanced Options
- **Cache Clearing**: Optional cache eviction between iterations for cold-cache benchmarks
- **Custom Metrics**: Define custom metric names for specialized benchmarks
- **Configurable Iterations**: Control over warmup iterations and measured iterations via `stage_config`
- **Programmatic Access**: Retrieve raw performance metrics via `get_test_results()`
- **Selective Metric Display**: Customize which metrics are shown in output

### Hardware Introspection
- **CPU Properties**: Comprehensive CPU detection and automatic reporting
- **GPU Properties**: CUDA device detection and reporting
- **Compiler Info**: Automatic compiler ID and version capture
- **OS Detection**: Operating system name and version in results

### Performance Counters
- **Cross-platform CPU counters**: Windows, Linux, macOS, Apple ARM
- **CUDA performance events**: GPU-specific performance monitoring

## API Conventions

As of v1.0.0, all APIs follow snake_case naming convention:
- Functions: `do_not_optimize_away()`, `generate_random_integers()`, `print_results()`
- Types: `size_type`, `string_literal`, `stage_config`
- Variables: `bytes_processed`, `test_values`

## Migrating from v1.0.0

If you're upgrading from v1.0.0 to v1.0.2:

### 1. Update stage_config usage

**Old:**
```cpp
bnch_swt::benchmark_stage<"test", 200, 25, bnch_swt::benchmark_types::cpu>
```

**New:**
```cpp
constexpr bnch_swt::stage_config config{
    .max_execution_count = 200,
    .measured_iteration_count = 25,
    .benchmark_type = bnch_swt::benchmark_types::cpu
};
bnch_swt::benchmark_stage<"test", config>
```

### 2. Add test name parameter to run_benchmark

**Old:**
```cpp
benchmark_stage::run_benchmark<"subject_name", function_type>(args...)
```

**New:**
```cpp
benchmark_stage::run_benchmark<"test_name", "subject_name", function_type>(args...)
```

### 3. Update run_from_host to run_benchmark_from_host

**Old:**
```cpp
benchmark_stage::run_from_host<"subject_name", function_type>(bytes_processed, args...)
```

**New:**
```cpp
benchmark_stage::run_benchmark_from_host<"test_name", "subject_name", function_type>(
    bytes_processed, args...)
```

### 4. Remove show_comparison parameter

**Old:**
```cpp
benchmark_stage::print_results(true, true)
```

**New:**
```cpp
benchmark_stage::print_results(true)  // show_metrics only
```

### 5. Update result access

**Old:**
```cpp
auto results = benchmark_stage::get_results();  // vector<performance_metrics>
```

**New:**
```cpp
auto all = benchmark_stage::get_all_results();           // vector<test_results>
auto test = benchmark_stage::get_test_results("test_name");  // unordered_map<string_view, metrics>
benchmark_stage::clear_all_results();  // New method
```

### 6. (Optional) Enable adaptive benchmarking features

```cpp
constexpr bnch_swt::stage_config adaptive_config{
    .max_execution_count = 10000,
    .measured_iteration_count = 50,
    .desired_percentage_deviation = 0.5,  // New: target stability
    .max_time_seconds = 10.0              // New: time limit
};
```

---

**Happy benchmarking with benchmarksuite v1.0.2!** 🚀

For issues, feature requests, or contributions, please visit the [GitHub repository](https://github.com/RealTimeChris/benchmarksuite).