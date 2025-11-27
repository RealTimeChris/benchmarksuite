# Benchmark Suite

Hello and welcome to bnch_swt or "Benchmark Suite". This is a collection of classes/functions for the purpose of benchmarking CPU performance.

The following operating systems and compilers are officially supported:

### Compiler Support
----
![MSVC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=MSVC&labelColor=pewter&color=blue&branch=main)
![GCC](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=GCC&labelColor=pewter&color=blue&branch=main)
![CLANG](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=CLANG&labelColor=pewter&color=blue&branch=main)

### Operating System Support
----
![Windows](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=microsoft&logoColor=green&label=Windows&labelColor=pewter&color=blue&branch=main)
![Linux](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=linux&logoColor=green&label=Linux&labelColor=pewter&color=blue&branch=main)
![Mac](https://img.shields.io/github/actions/workflow/status/RealTimeChris/benchmarksuite/benchmark.yml?style=plastic&logo=apple&logoColor=green&label=MacOS&labelColor=pewter&color=blue&branch=main)

# Quickstart Guide for benchmarksuite

This guide will walk you through setting up and running benchmarks using `benchmarksuite`.

## Table of Contents
- [Installation](#installation)
- [Basic Example](#basic-example)
- [Creating Benchmarks](#creating-benchmarks)
- [Running Benchmarks](#running-benchmarks)
- [Output and Results](#output-and-results)

## Installation
To use `benchmarksuite`, include the necessary header files in your project. Ensure you have a C++23 (or later) compliant compiler.

```cpp
#include <bnch_swt/benchmarksuite.hpp>
#include <vector>
#include <string>
#include <cstring>
```

## Basic Example
The following example demonstrates how to set up and run a benchmark comparing two integer-to-string conversion functions:

```cpp
template<uint64_t count, typename value_type, bnch_swt::string_literal testName>
BNCH_SWT_HOST void testFunction() {
    std::vector<value_type> testValues{ generateRandomIntegers<value_type>(count, sizeof(value_type) == 4 ? 10 : 20) };
    std::vector<std::string> testValues00;
    std::vector<std::string> testValues01(count);

    for (uint64_t x = 0; x < count; ++x) {
        testValues00.emplace_back(std::to_string(testValues[x]));
    }

    bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::template run_benchmark<"glz::to_chars", "CYAN">([&] {
        uint64_t bytesProcessed = 0;
        char newerString[30]{};
        for (uint64_t x = 0; x < count; ++x) {
            std::memset(newerString, '\0', sizeof(newerString));
            auto newPtr = to_chars(newerString, testValues[x]);
            bytesProcessed += testValues00[x].size();
            testValues01[x] = std::string{newerString, static_cast<uint64_t>(newPtr - newerString)};
        }
        bnch_swt::doNotOptimizeAway(bytesProcessed);
        return bytesProcessed;
    });

    bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::template run_benchmark<"jsonifier_internal::toChars", "CYAN">([&] {
        uint64_t bytesProcessed = 0;
        char newerString[30]{};
        for (uint64_t x = 0; x < count; ++x) {
            std::memset(newerString, '\0', sizeof(newerString));
            auto newPtr = jsonifier_internal::toChars(newerString, testValues[x]);
            bytesProcessed += testValues00[x].size();
            testValues01[x] = std::string{newerString, static_cast<uint64_t>(newPtr - newerString)};
        }
        bnch_swt::doNotOptimizeAway(bytesProcessed);
        return bytesProcessed;
    });

    bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::printResults(true, false);
}

int main() {
    testFunction<512, uint64_t, "-uint64">();
    testFunction<512, int64_t, "-int64">();
    return 0;
}
```

## Creating Benchmarks
To create a benchmark:
1. Generate or initialize test data.
2. Use `bnch_swt::benchmark_stage` to define a benchmark. By setting the name of the `bnch_swt::benchmark_stage` using a string literal, you are instantiating a single "stage" within which to execute different benchmarks.
3. Implement test functions with lambdas capturing your benchmark logic.

### Benchmark Stage
The `benchmark_stage` structure orchestrates each test:

### Methods
- `run_benchmark()`: Executes a given lambda function, measuring performance. By setting the name of the benchmark 'run' using a string literal, you are instantiating a single benchmark "entity" or "library" to have its data collected and compared, within the given benchmark stage.
- `printResults()`: Displays detailed performance metrics and comparisons.

### Example Benchmark Definitions
- **`run_benchmark`**: Executes a lambda function and tracks performance.
  - `"glz::to_chars"`: A label for the function being benchmarked.
  - `"jsonifier_internal::toChars"`: An alternative implementation to compare.

### Avoiding Compiler Optimizations
Use `bnch_swt::doNotOptimizeAway` to prevent the compiler from optimizing away results.

## Running Benchmarks
Compile and run your program:

## Output and Results
```c
Performance Metrics for: int-to-string-comparisons-1
Metrics for: benchmarksuite::internal::toChars
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
Library benchmarksuite::internal::toChars, is faster than library: glz::to_chars, by roughly: 11.36%.
```

This structured output helps you quickly identify which implementation is faster or more efficient.

---

Now you’re ready to start benchmarking with **benchmarksuite**!

