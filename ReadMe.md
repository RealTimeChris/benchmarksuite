# Benchmark Suite

Hello and welcome to bnch_swt or "Benchmark Suite". This is a collection of classes/functions for the purpose of benchmarking CPU and GPU performance.

The following operating systems and compilers are officially supported:

### Compiler Support
----
![MSVC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=MSVC&labelColor=pewter&color=blue&branch=main)
![GCC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=GCC&labelColor=pewter&color=blue&branch=main)
![CLANG](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=CLANG&labelColor=pewter&color=blue&branch=main)
![NVCC](https://img.shields.io/badge/NVCC-Supported-blue?style=plastic&logo=nvidia&logoColor=green&labelColor=pewter)

### Operating System Support
----
![Windows](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=Windows&labelColor=pewter&color=blue&branch=main)
![Linux](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=Linux&labelColor=pewter&color=blue&branch=main)
![Mac](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=MacOS&labelColor=pewter&color=blue&branch=main)

# Quickstart Guide for benchmarksuite

This guide will walk you through setting up and running benchmarks using `benchmarksuite`.

## Table of Contents
- [Installation](#installation)
  - [Method 1: vcpkg + CMake (Recommended)](#method-1-vcpkg--cmake-recommended)
  - [Method 2: Manual Installation](#method-2-manual-installation)
  - [Requirements](#requirements)
- [Basic Example](#basic-example)
- [Creating Benchmarks](#creating-benchmarks)
- [CPU vs GPU Benchmarking](#cpu-vs-gpu-benchmarking)
- [Advanced Benchmark Methods](#advanced-benchmark-methods)
- [Running Benchmarks](#running-benchmarks)
  - [Common CMake Options](#common-cmake-options)
  - [Complete Project Example](#complete-project-example)
- [Output and Results](#output-and-results)
- [Features](#features)
- [API Conventions](#api-conventions)
- [Migrating from Pre-1.0.0](#migrating-from-pre-100)

## Installation

### Method 1: vcpkg + CMake (Recommended)

**Step 1: Add to vcpkg.json**

Create or update your `vcpkg.json` in your project root:

```json
{
  "name": "your-project-name",
  "version": "1.0.0",
  "dependencies": [
    "benchmarksuite"
  ]
}
```

**Step 2: Configure CMake**

In your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(YourProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
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

To use `benchmarksuite`, ensure you have a C++23 (or later) compliant compiler.

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
    constexpr uint64_t count = 512;
    
    std::vector<int64_t> test_values = generate_random_integers<int64_t>(count, 20);
    std::vector<std::string> test_values_00;
    std::vector<std::string> test_values_01(count);
    
    for (uint64_t x = 0; x < count; ++x) {
        test_values_00.emplace_back(std::to_string(test_values[x]));
    }
    
    using benchmark = bnch_swt::benchmark_stage<"int-to-string-comparison", 200, 25, 
                                                 bnch_swt::benchmark_types::cpu>;
    
    benchmark::run_benchmark<"glz::to_chars", glz_to_chars_benchmark>(test_values, test_values_00, test_values_01);
    benchmark::run_benchmark<"jsonifier::to_chars", jsonifier_to_chars_benchmark>(test_values, test_values_00, test_values_01);
    
    benchmark::print_results(true, true);
    
    return 0;
}
```

## Creating Benchmarks
To create a benchmark:
1. Define your benchmark functions as structs with a static `impl()` method that returns `uint64_t` (bytes processed)
2. Use `bnch_swt::benchmark_stage` with appropriate template parameters
3. Call `run_benchmark` with your benchmark struct and any required arguments

### Benchmark Stage
The `benchmark_stage` structure orchestrates each test and supports both CPU and GPU benchmarking:

```cpp
template<bnch_swt::string_literal stage_name,
         uint64_t max_execution_count = 200,
         uint64_t measured_iteration_count = 25,
         bnch_swt::benchmark_types benchmark_type = bnch_swt::benchmark_types::cpu,
         bool clear_cpu_cache_between_each_iteration = false,
         bnch_swt::string_literal metric_name = bnch_swt::string_literal<1>{}
>
struct benchmark_stage;

using cpu_benchmark = bnch_swt::benchmark_stage<"my-benchmark">;
using gpu_benchmark = bnch_swt::benchmark_stage<"gpu-test", 100, 10, bnch_swt::benchmark_types::cuda>;
using custom_metric = bnch_swt::benchmark_stage<"compression", 200, 25, bnch_swt::benchmark_types::cpu, false, "compression-ratio">;
```

### Template Parameters
- **stage_name** (required): String literal identifying the benchmark stage
- **max_execution_count** (default 200): Total number of iterations including warmup
- **measured_iteration_count** (default 25): Number of iterations to measure for final metrics
- **benchmark_type** (default cpu): `bnch_swt::benchmark_types::cpu` or `bnch_swt::benchmark_types::cuda`
- **clear_cpu_cache_between_each_iteration** (default false): Whether to clear CPU caches between iterations
- **metric_name** (default empty): Custom metric name for specialized benchmarks (e.g., compression ratios)

### Methods

#### `run_benchmark<name, function_type>(args...)`
Executes the benchmark using a struct with a static `impl()` method.

**Parameters:**
- **name**: String literal identifying this specific benchmark within the stage
- **function_type**: Struct type with a static `impl()` method
- **args...**: Arguments forwarded to the `impl()` method

**Returns:** `performance_metrics<benchmark_type>` object

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

using bench = bnch_swt::benchmark_stage<"test">;
std::vector<int> data(1000);
bench::run_benchmark<"my-test", my_benchmark>(data);
```

#### `run_benchmark<name, function>(args...)`
Executes the benchmark using a function or lambda directly (passed as non-type template parameter).

**Parameters:**
- **name**: String literal identifying this specific benchmark
- **function**: Function or lambda to benchmark (as non-type template parameter)
- **args...**: Arguments forwarded to the function

**Returns:** `performance_metrics<benchmark_type>` object

**Example:**
```cpp
constexpr auto my_lambda = [](std::vector<int>& data) -> uint64_t {
    uint64_t sum = 0;
    for (auto& val : data) {
        sum += val;
    }
    return data.size() * sizeof(int);
};

using bench = bnch_swt::benchmark_stage<"test">;
std::vector<int> data(1000);
bench::run_benchmark<"my-test", my_lambda>(data);
```

#### `run_from_host<name, function>(args...)`
Executes the benchmark from the host (useful for CUDA kernels launched from host code).

**Parameters:**
- **name**: String literal identifying this specific benchmark
- **function**: Function type to benchmark
- **args...**: Arguments forwarded to the function

**Returns:** `performance_metrics<benchmark_type>` object

**Example:**
```cpp
struct cuda_host_launcher {
    static uint64_t impl(float* gpu_data, uint64_t size) {
        dim3 grid{256};
        dim3 block{256};
        my_kernel<<<grid, block>>>(gpu_data, size);
        cudaDeviceSynchronize();
        return size * sizeof(float);
    }
};

using bench = bnch_swt::benchmark_stage<"cuda-test", 100, 10, bnch_swt::benchmark_types::cuda>;
float* gpu_data;
cudaMalloc(&gpu_data, 1024 * sizeof(float));
bench::run_from_host<"kernel-test", cuda_host_launcher>(gpu_data, 1024);
```

#### `run_benchmark_cooperative<name, function>(args...)`
Executes the benchmark using CUDA cooperative groups (for kernels requiring grid-wide synchronization).

**Parameters:**
- **name**: String literal identifying this specific benchmark
- **function**: Function to benchmark (as non-type template parameter)
- **args...**: Arguments forwarded to the function

**Returns:** `performance_metrics<benchmark_type>` object

**Example:**
```cpp
constexpr auto cooperative_kernel = [](float* data, uint64_t size) -> uint64_t {
    return size * sizeof(float);
};

using bench = bnch_swt::benchmark_stage<"cooperative-test", 100, 10, bnch_swt::benchmark_types::cuda>;
float* gpu_data;
cudaMalloc(&gpu_data, 1024 * sizeof(float));
bench::run_benchmark_cooperative<"coop-kernel", cooperative_kernel>(gpu_data, 1024);
```

#### `print_results(show_comparison = true, show_metrics = true)`
Displays performance metrics and comparisons.

**Parameters:**
- **show_comparison**: Whether to show head-to-head comparisons between benchmarks
- **show_metrics**: Whether to show detailed hardware counter metrics

**Example:**
```cpp
benchmark::print_results(true, true);
```

You can also customize which metrics are displayed:

```cpp
bnch_swt::performance_metrics_presence<bnch_swt::benchmark_types::cpu> custom_metrics{};
custom_metrics.throughput_mb_per_sec = true;
custom_metrics.cycles_per_byte = true;
custom_metrics.instructions_per_cycle = true;
benchmark::print_results<custom_metrics>(true, true);
```

#### `get_results()`
Returns a sorted vector of all `performance_metrics` for programmatic access.

**Returns:** `std::vector<performance_metrics<benchmark_type>>`

**Example:**
```cpp
auto results = benchmark::get_results();
for (const auto& metric : results) {
    std::cout << metric.name << ": " << metric.throughput_mb_per_sec << " MB/s\n";
}
```

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
    }
};
```

**Key differences:**
- **CPU**: `impl()` returns `uint64_t` (bytes processed) and uses `BNCH_SWT_HOST`
- **CUDA**: `impl()` returns `void`, uses `BNCH_SWT_DEVICE`, and contains kernel code
- **CUDA**: Bytes processed is passed as a parameter to `run_benchmark()`, not returned from `impl()`

## CPU vs GPU Benchmarking

As of v1.0.0, benchmarksuite supports both CPU and GPU (CUDA) benchmarking through the `benchmark_types` enum.

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

using cpu_stage = bnch_swt::benchmark_stage<"cpu-test", 200, 25, bnch_swt::benchmark_types::cpu>;

constexpr size_t data_size = 1024 * 1024;
std::vector<float> input(data_size, 1.0f);
std::vector<float> output(data_size);

cpu_stage::run_benchmark<"my-cpu-function", cpu_computation_benchmark>(input, output);

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

using cuda_stage = bnch_swt::benchmark_stage<"gpu-test", 100, 10, bnch_swt::benchmark_types::cuda>;

constexpr uint64_t data_size = 1024 * 1024;
float* gpu_data;
cudaMalloc(&gpu_data, data_size * sizeof(float));

dim3 grid{256, 1, 1};
dim3 block{256, 1, 1};
uint64_t shared_memory = 0;
uint64_t bytes_processed = data_size * sizeof(float);

cuda_stage::run_benchmark<"my-cuda-kernel", cuda_kernel_benchmark>(
    grid, block, shared_memory, bytes_processed, 
    gpu_data, data_size
);

cuda_stage::print_results();
cudaFree(gpu_data);
```

### Mixed CPU/GPU Benchmarking
You can benchmark CPU and GPU implementations side-by-side:

```cpp
constexpr uint64_t data_size = 1024 * 1024;

struct cpu_process_benchmark {
    BNCH_SWT_HOST static uint64_t impl(std::vector<float>& cpu_data) {
        for (size_t i = 0; i < cpu_data.size(); ++i) {
            cpu_data[i] = cpu_data[i] * 2.0f;
        }
        return cpu_data.size() * sizeof(float);
    }
};

struct gpu_process_benchmark {
    BNCH_SWT_DEVICE static void impl(float* gpu_data, uint64_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            gpu_data[idx] = gpu_data[idx] * 2.0f;
        }
    }
};

std::vector<float> cpu_data(data_size);
float* gpu_data;
cudaMalloc(&gpu_data, data_size * sizeof(float));

using cpu_test = bnch_swt::benchmark_stage<"cpu-vs-gpu", 100, 10, bnch_swt::benchmark_types::cpu>;
cpu_test::run_benchmark<"cpu-version", cpu_process_benchmark>(cpu_data);

using gpu_test = bnch_swt::benchmark_stage<"cpu-vs-gpu", 100, 10, bnch_swt::benchmark_types::cuda>;
dim3 grid{(data_size + 255) / 256, 1, 1};
dim3 block{256, 1, 1};
gpu_test::run_benchmark<"gpu-version", gpu_process_benchmark>(
    grid, block, 0, data_size * sizeof(float),
    gpu_data, data_size
);

cpu_test::print_results();
gpu_test::print_results();

cudaFree(gpu_data);
```

### Cache Clearing Option
For more accurate CPU benchmarks, you can enable cache clearing between iterations:

```cpp
using cache_cleared = bnch_swt::benchmark_stage<"cache-test", 200, 25, bnch_swt::benchmark_types::cpu, true>;
```

This is useful when benchmarking memory-bound operations where you want to measure cold cache performance.

### Custom Metrics
You can specify custom metric names for specialized benchmarks that don't measure traditional throughput:

```cpp
using compression_bench = bnch_swt::benchmark_stage<"compression-test", 200, 25, 
                                                     bnch_swt::benchmark_types::cpu, 
                                                     false, 
                                                     "compression-ratio">;

struct compress_benchmark {
    BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& input) {
        auto compressed = compress_data(input);
        return (input.size() * 1000) / compressed.size();
    }
};

compression_bench::run_benchmark<"my-compressor", compress_benchmark>(input_data);
compression_bench::print_results();
```

When a custom metric name is provided, the results will display your custom metric instead of standard MB/s throughput.

## Advanced Benchmark Methods

### Host-Launched Kernels
Use `run_from_host()` when you need to launch CUDA kernels from host code with custom configurations:

```cpp
struct custom_kernel_launcher {
    static uint64_t impl(float* data, uint64_t size, int custom_param) {
        dim3 grid{static_cast<unsigned int>((size + 255) / 256)};
        dim3 block{256};
        size_t shared_mem = custom_param * sizeof(float);
        
        my_kernel<<<grid, block, shared_mem>>>(data, size);
        cudaDeviceSynchronize();
        
        return size * sizeof(float);
    }
};

using bench = bnch_swt::benchmark_stage<"custom-kernel", 100, 10, bnch_swt::benchmark_types::cuda>;
float* gpu_data;
cudaMalloc(&gpu_data, 1024 * sizeof(float));
bench::run_from_host<"custom-launch", custom_kernel_launcher>(gpu_data, 1024, 32);
```

### Cooperative Kernels
Use `run_benchmark_cooperative()` for kernels that require grid-wide synchronization:

```cpp
constexpr auto cooperative_reduce = [](float* data, float* result, uint64_t size) -> uint64_t {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    __shared__ float shared_data[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared_data[threadIdx.x] = (idx < size) ? data[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(result, shared_data[0]);
    }
    
    grid.sync();
    
    return size * sizeof(float);
};

using bench = bnch_swt::benchmark_stage<"cooperative-test", 100, 10, bnch_swt::benchmark_types::cuda>;
float* gpu_data;
float* gpu_result;
cudaMalloc(&gpu_data, 1024 * sizeof(float));
cudaMalloc(&gpu_result, sizeof(float));
bench::run_benchmark_cooperative<"grid-reduce", cooperative_reduce>(gpu_data, gpu_result, 1024);
```

### Function vs Struct Benchmarks
You can use either approach depending on your needs:

**Struct-based (recommended for complex benchmarks):**
```cpp
struct complex_benchmark {
    BNCH_SWT_HOST static uint64_t impl(std::vector<int>& data, int multiplier) {
        for (auto& val : data) {
            val *= multiplier;
        }
        return data.size() * sizeof(int);
    }
};

bench::run_benchmark<"complex", complex_benchmark>(data, 2);
```

**Function-based (convenient for simple benchmarks):**
```cpp
constexpr auto simple_benchmark = [](std::vector<int>& data, int multiplier) -> uint64_t {
    for (auto& val : data) {
        val *= multiplier;
    }
    return data.size() * sizeof(int);
};

bench::run_benchmark<"simple", simple_benchmark>(data, 2);
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

**For CUDA benchmarks, ensure CUDA is enabled:**

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

set(CMAKE_CXX_STANDARD 23)
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
    "benchmarksuite"
  ]
}
```

## Output and Results
```
CPU Performance Metrics for: int-to-string-comparisons-1
Metrics for: benchmarksuite::internal::to_chars
Total Iterations to Stabilize                               : 394
Measured Iterations                                         : 20
Bytes Processed                                             : 512.00
Nanoseconds per Execution                                   : 5785.25
Frequency (GHz)                                             : 4.83
Throughput (MB/s)                                           : 84.58
Throughput Percentage Deviation (+/-%)                      : 8.36
Cycles per Execution                                        : 27921.20
Cycles per Byte                                             : 54.53
Instructions per Execution                                  : 52026.00
Instructions per Cycle                                      : 1.86
Instructions per Byte                                       : 101.61
Branches per Execution                                      : 361.45
Branch Misses per Execution                                 : 0.73
Cache References per Execution                              : 97.03
Cache Misses per Execution                                  : 74.68
----------------------------------------
Metrics for: glz::to_chars
Total Iterations to Stabilize                               : 421
Measured Iterations                                         : 20
Bytes Processed                                             : 512.00
Nanoseconds per Execution                                   : 6480.30
Frequency (GHz)                                             : 4.68
Throughput (MB/s)                                           : 75.95
Throughput Percentage Deviation (+/-%)                      : 17.58
Cycles per Execution                                        : 30314.40
Cycles per Byte                                             : 59.21
Instructions per Execution                                  : 51513.00
Instructions per Cycle                                      : 1.70
Instructions per Byte                                       : 100.61
Branches per Execution                                      : 438.25
Branch Misses per Execution                                 : 0.73
Cache References per Execution                              : 95.93
Cache Misses per Execution                                  : 73.59
----------------------------------------
Library benchmarksuite::internal::to_chars is faster than library glz::to_chars by 11.36%.
```

This structured output helps you quickly identify which implementation is faster or more efficient.

## Features

### Dual Benchmarking Support
- **CPU Benchmarking**: Traditional CPU performance measurement with hardware counters
- **GPU/CUDA Benchmarking**: Native CUDA kernel benchmarking with grid/block configuration
- **Mixed Workloads**: Compare CPU vs GPU implementations side-by-side
- **Automatic Device Selection**: Choose benchmark type via `bnch_swt::benchmark_types::cpu` or `bnch_swt::benchmark_types::cuda`

### Advanced Execution Modes
- **Standard Benchmarking**: Default `run_benchmark()` for most use cases
- **Host-Launched Kernels**: `run_from_host()` for custom kernel launch configurations
- **Cooperative Groups**: `run_benchmark_cooperative()` for grid-wide synchronization
- **Function or Struct**: Support for both function-based and struct-based benchmarks

### Advanced Options
- **Cache Clearing**: Optional cache eviction between iterations for cold-cache benchmarks
- **Custom Metrics**: Define custom metric names for specialized benchmarks (e.g., compression ratios, custom throughput units)
- **Configurable Iterations**: Separate control over warmup iterations and measured iterations
- **Programmatic Access**: Retrieve raw performance metrics via `get_results()` for custom analysis
- **Selective Metric Display**: Customize which metrics are shown in output

### Hardware Introspection
- **CPU Properties**: Comprehensive CPU detection and properties via `benchmarksuite_cpu_properties.hpp`
- **GPU Properties**: CUDA device detection and properties via `benchmarksuite_gpu_properties.hpp`

### Performance Counters
- **Cross-platform CPU counters**: Windows, Linux, macOS, Android, Apple ARM
- **CUDA performance events**: GPU-specific performance monitoring via `counters/cuda_perf_events.hpp`

### Utilities
- **Cache management**: Cross-platform cache clearing utilities
- **Aligned constants**: Compile-time aligned data structures
- **Random generators**: High-quality random data generation for benchmarks

## API Conventions

As of v1.0.0, all APIs follow snake_case naming convention:
- Functions: `do_not_optimize_away()`, `generate_random_integers()`, `print_results()`
- Types: `size_type`, `string_literal`
- Variables: `bytes_processed`, `test_values`

## Migrating from Pre-1.0.0

If you're upgrading from an earlier version:

1. **Update package name**: Keep using `benchmarksuite`

2. **Update include paths**: All includes are lowercase (already standard)

3. **Update API calls**: Convert camelCase/PascalCase to snake_case
   - `doNotOptimizeAway()` → `do_not_optimize_away()`
   - `printResults()` → `print_results()`
   - `generateRandomIntegers()` → `generate_random_integers()`

4. **Change benchmark interface**: Lambdas are replaced with structs (or use function template parameter)
   ```cpp
   benchmark_stage<"test">::run_benchmark<"name">([&] {
       return bytes_processed;
   });
   
   struct my_benchmark {
       BNCH_SWT_HOST static uint64_t impl(/* params */) {
           return bytes_processed;
       }
   };
   benchmark_stage<"test">::run_benchmark<"name", my_benchmark>(/* args */);
   
   constexpr auto my_lambda = [](/* params */) -> uint64_t {
       return bytes_processed;
   };
   benchmark_stage<"test">::run_benchmark<"name", my_lambda>(/* args */);
   ```

5. **Update template parameters**: benchmark_stage now has more options
   ```cpp
   benchmark_stage<"test", iterations, measured>
   
   benchmark_stage<"test", 200, 25, benchmark_types::cpu, false, "">
   ```

6. **New feature - Device types**: You can now specify CPU or CUDA benchmarking:
   ```cpp
   benchmark_stage<"test", 200, 25, bnch_swt::benchmark_types::cpu>
   
   benchmark_stage<"test", 100, 10, bnch_swt::benchmark_types::cuda>
   ```

7. **New feature - Cache clearing**: Enable cache clearing between iterations for CPU benchmarks:
   ```cpp
   benchmark_stage<"test", 200, 25, benchmark_types::cpu, true>
   ```

8. **New feature - Custom metrics**: Specify custom metric names for specialized benchmarks:
   ```cpp
   benchmark_stage<"compression-test", 200, 25, benchmark_types::cpu, false, "compression-ratio">
   ```

9. **New feature - Advanced execution modes**: Additional methods for specialized use cases:
   ```cpp
   benchmark_stage::run_from_host<"name", function>(args...);
   benchmark_stage::run_benchmark_cooperative<"name", function>(args...);
   ```

---

Now you're ready to start benchmarking with **benchmarksuite**!