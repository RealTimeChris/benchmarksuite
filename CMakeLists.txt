#	MIT License
#
#	Copyright (c) 2024 RealTimeChris
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy of this
#	software and associated documentation files (the "Software"), to deal in the Software
#	without restriction, including without limitation the rights to use, copy, modify, merge,
#	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#	persons to whom the Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all copies or
#	substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#	DEALINGS IN THE SOFTWARE.
cmake_minimum_required(VERSION 3.28)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(BNCH_SWT_DETECT_CPU_PROPERTIES "Override cpu-cache-size selection" OFF)
option(BNCH_SWT_DETECT_GPU_PROPERTIES "Override cuda-cache-size selection" OFF)
option(BNCH_SWT_CUDA "Enable CUDA support" OFF)

include(cmake/get_version.cmake)

if(BNCH_SWT_CUDA)
    set(BNCH_SWT_LANGUAGE CUDA)
    project(benchmarksuite
        VERSION ${CLEAN_VERSION}
        DESCRIPTION "A header-only C++ benchmarking library with cross-platform hardware performance counter integration, providing precise measurements of cycles, instructions, branches, cache behavior, and throughput with minimal overhead."
        LANGUAGES CUDA CXX
    )
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
    include(cmake/detection/benchmarksuite_detect_gpu_properties.cmake)
    set(CMAKE_CUDA_ARCHITECTURES ${BNCH_SWT_MAJOR_COMPUTE_CAPABILITY}${BNCH_SWT_MINOR_COMPUTE_CAPABILITY} CACHE STRING "Cuda architectures." FORCE)
    set(BNCH_SWT_MAIN_FILE_EXTENSION .cu)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(BNCH_SWT_LANGUAGE CXX)
    project(benchmarksuite
        VERSION ${CLEAN_VERSION}
        DESCRIPTION "A header-only C++ benchmarking library with cross-platform hardware performance counter integration, providing precise measurements of cycles, instructions, branches, cache behavior, and throughput with minimal overhead."
        LANGUAGES CXX
    )
    set(BNCH_SWT_MAIN_FILE_EXTENSION .cpp)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/detection/benchmarksuite_detect_cpu_properties.cmake)
option(CHUNKS_PER_BLOCK "" 2)
option(BLOCKS_PER_STEP "" 8)
include(cmake/library_setup.cmake)

include(cmake/installation_setup.cmake)

message(STATUS "benchmarksuite Configuration Summary:")
message(STATUS "=============================")
message(STATUS "Library Version: ${CLEAN_VERSION}")
message(STATUS "Max Thread Count: ${BNCH_SWT_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${BNCH_SWT_COMPILER_ID}")
message(STATUS "CPU Variants: ${BNCH_SWT_INSTRUCTION_SET_NAME}")

if(BNCH_SWT_CUDA)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(BNCH_SWT_BENCHMARKS)
    add_subdirectory(src)
endif()
if(BNCH_SWT_UNIT_TESTS)
    add_subdirectory(unit-tests)
endif()