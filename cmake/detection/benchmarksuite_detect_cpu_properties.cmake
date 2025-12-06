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

if(UNIX OR APPLE)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_cpu_properties.sh "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cpu-Properties -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\" -DBNCH_SWT_DETECT_CPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cpu-Properties --config=Release"
    )
    
    execute_process(
        COMMAND chmod +x ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_cpu_properties.sh
        RESULT_VARIABLE CHMOD_RESULT
    )
    
    if(NOT CHMOD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for build_feature_tester_cpu_properties.sh")
    endif()
    
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_cpu_properties.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection
    )
    
    set(FEATURE_TESTER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/Build-Cpu-Properties/feature_detector)
    
elseif(WIN32)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_cpu_properties.bat
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cpu-Properties -DBNCH_SWT_DETECT_CPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cpu-Properties --config=Release"
    )
    
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_cpu_properties.bat
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection
    )
    
    set(FEATURE_TESTER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/Build-Cpu-Properties/Release/feature_detector.exe)
endif()

if(NOT DEFINED BNCH_SWT_THREAD_COUNT OR
   NOT DEFINED BNCH_SWT_CPU_L1_CACHE_SIZE OR
   NOT DEFINED BNCH_SWT_CPU_L2_CACHE_SIZE OR
   NOT DEFINED BNCH_SWT_CPU_L3_CACHE_SIZE OR
   NOT BNCH_SWT_DETECT_CPU_PROPERTIES)
    
    execute_process(
        COMMAND ${FEATURE_TESTER_FILE}
        RESULT_VARIABLE FEATURE_TESTER_EXIT_CODE
        OUTPUT_VARIABLE CPU_PROPERTIES_OUTPUT
        ERROR_VARIABLE FEATURE_TESTER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

message(STATUS "CPU detector exit code: ${FEATURE_TESTER_EXIT_CODE}")
message(STATUS "CPU detector output: '${CPU_PROPERTIES_OUTPUT}'")
message(STATUS "CPU detector error: '${FEATURE_TESTER_ERROR}'")

if(FEATURE_TESTER_EXIT_CODE EQUAL 0 AND CPU_PROPERTIES_OUTPUT MATCHES "CPU_SUCCESS=1")
    
    string(REGEX MATCH "THREAD_COUNT=([0-9]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_THREAD_COUNT)
        set(BNCH_SWT_THREAD_COUNT ${CMAKE_MATCH_1} CACHE STRING "CPU thread count" FORCE)
    endif()
    
    string(REGEX MATCH "INSTRUCTION_SET=([0-9]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_INSTRUCTION_SET)
        set(BNCH_SWT_INSTRUCTION_SET ${CMAKE_MATCH_1} CACHE STRING "CPU instruction set bitmask" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_AVX2=([0-1]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_HAS_AVX2)
        set(BNCH_SWT_HAS_AVX2 ${CMAKE_MATCH_1} CACHE STRING "CPU has AVX2 support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_AVX512=([0-1]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_HAS_AVX512)
        set(BNCH_SWT_HAS_AVX512 ${CMAKE_MATCH_1} CACHE STRING "CPU has AVX512 support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_NEON=([0-1]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_HAS_NEON)
        set(BNCH_SWT_HAS_NEON ${CMAKE_MATCH_1} CACHE STRING "CPU has NEON support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_SVE2=([0-1]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_HAS_SVE2)
        set(BNCH_SWT_HAS_SVE2 ${CMAKE_MATCH_1} CACHE STRING "CPU has SVE2 support" FORCE)
    endif()
    
    string(REGEX MATCH "L1_CACHE_SIZE=([0-9]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_CPU_L1_CACHE_SIZE)
        set(BNCH_SWT_CPU_L1_CACHE_SIZE ${CMAKE_MATCH_1} CACHE STRING "CPU L1 cache size" FORCE)
    endif()
    
    string(REGEX MATCH "L2_CACHE_SIZE=([0-9]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_CPU_L2_CACHE_SIZE)
        set(BNCH_SWT_CPU_L2_CACHE_SIZE ${CMAKE_MATCH_1} CACHE STRING "CPU L2 cache size" FORCE)
    endif()
    
    string(REGEX MATCH "L3_CACHE_SIZE=([0-9]+)" _ ${CPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_CPU_L3_CACHE_SIZE)
        set(BNCH_SWT_CPU_L3_CACHE_SIZE ${CMAKE_MATCH_1} CACHE STRING "CPU L3 cache size" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_CPU_PROPERTIES_ERECTED)
        set(BNCH_SWT_CPU_PROPERTIES_ERECTED TRUE CACHE BOOL "CPU properties successfully detected" FORCE)
    endif()
    
    message(STATUS "CPU Properties detected successfully")
    
else()
    message(WARNING "CPU feature detector failed, using reasonable default values for unset properties")
    
    if(NOT DEFINED BNCH_SWT_THREAD_COUNT)
        set(BNCH_SWT_THREAD_COUNT 4 CACHE STRING "CPU thread count (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_INSTRUCTION_SET)
        set(BNCH_SWT_INSTRUCTION_SET 0 CACHE STRING "CPU instruction set bitmask (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_HAS_AVX2)
        set(BNCH_SWT_HAS_AVX2 0 CACHE STRING "CPU has AVX2 support (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_HAS_AVX512)
        set(BNCH_SWT_HAS_AVX512 0 CACHE STRING "CPU has AVX512 support (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_HAS_NEON)
        set(BNCH_SWT_HAS_NEON 0 CACHE STRING "CPU has NEON support (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_HAS_SVE2)
        set(BNCH_SWT_HAS_SVE2 0 CACHE STRING "CPU has SVE2 support (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_CPU_L1_CACHE_SIZE)
        set(BNCH_SWT_CPU_L1_CACHE_SIZE 32768 CACHE STRING "CPU L1 cache size - 32KB (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_CPU_L2_CACHE_SIZE)
        set(BNCH_SWT_CPU_L2_CACHE_SIZE 262144 CACHE STRING "CPU L2 cache size - 256KB (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_CPU_L3_CACHE_SIZE)
        set(BNCH_SWT_CPU_L3_CACHE_SIZE 8388608 CACHE STRING "CPU L3 cache size - 8MB (fallback)" FORCE)
    endif()
endif()

if(NOT DEFINED BNCH_SWT_CPU_ARCH_INDEX)
    if(BNCH_SWT_HAS_AVX512)
        set(BNCH_SWT_CPU_ARCH_INDEX 2 CACHE STRING "CPU architecture index - AVX512" FORCE)
        set(BNCH_SWT_CPU_ALIGNMENT 64 CACHE STRING "CPU Alignment" FORCE)
        set(BNCH_SWT_SIMD_FLAGS $<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX512,-mavx512f;-mavx512bw;-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>> CACHE STRING "SIMD flags" FORCE)
        set(BNCH_SWT_SIMD_DEFINITIONS BNCH_SWT_SVE2=0;BNCH_SWT_AVX512=1;BNCH_SWT_AVX2=0;BNCH_SWT_NEON=0 CACHE STRING "SIMD definitions" FORCE)
        set(BNCH_SWT_INSTRUCTION_SET_NAME AVX512 CACHE STRING "Instruction set name" FORCE)        
    elseif(BNCH_SWT_HAS_AVX2)
        set(BNCH_SWT_CPU_ARCH_INDEX 1 CACHE STRING "CPU architecture index - AVX2" FORCE)
        set(BNCH_SWT_CPU_ALIGNMENT 32 CACHE STRING "CPU Alignment" FORCE)
        set(BNCH_SWT_SIMD_FLAGS $<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2;-mfma;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>> CACHE STRING "SIMD flags" FORCE)
        set(BNCH_SWT_SIMD_DEFINITIONS BNCH_SWT_SVE2=0;BNCH_SWT_AVX512=0;BNCH_SWT_AVX2=1;BNCH_SWT_NEON=0 CACHE STRING "SIMD definitions" FORCE)
        set(BNCH_SWT_INSTRUCTION_SET_NAME AVX2 CACHE STRING "Instruction set name" FORCE)        
    elseif(BNCH_SWT_HAS_SVE2)
        set(BNCH_SWT_CPU_ARCH_INDEX 2 CACHE STRING "CPU architecture index - SVE2" FORCE)
        set(BNCH_SWT_CPU_ALIGNMENT 64 CACHE STRING "CPU Alignment" FORCE)
        set(BNCH_SWT_SIMD_FLAGS $<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a+sve;-msve-vector-bits=scalable;-march=armv8-a+sve+sve2>> CACHE STRING "SIMD flags" FORCE)
        set(BNCH_SWT_SIMD_DEFINITIONS BNCH_SWT_SVE2=1;BNCH_SWT_AVX512=0;BNCH_SWT_AVX2=0;BNCH_SWT_NEON=0 CACHE STRING "SIMD definitions" FORCE)
        set(BNCH_SWT_INSTRUCTION_SET_NAME SVE2 CACHE STRING "Instruction set name" FORCE)        
    elseif(BNCH_SWT_HAS_NEON)
        set(BNCH_SWT_CPU_ARCH_INDEX 1 CACHE STRING "CPU architecture index - NEON" FORCE)
        set(BNCH_SWT_CPU_ALIGNMENT 16 CACHE STRING "CPU Alignment" FORCE)
        set(BNCH_SWT_SIMD_FLAGS $<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a>> CACHE STRING "SIMD flags" FORCE)
        set(BNCH_SWT_SIMD_DEFINITIONS BNCH_SWT_SVE2=0;BNCH_SWT_AVX512=0;BNCH_SWT_AVX2=0;BNCH_SWT_NEON=1 CACHE STRING "SIMD definitions" FORCE)
        set(BNCH_SWT_INSTRUCTION_SET_NAME NEON CACHE STRING "Instruction set name" FORCE)        
    else()
        set(BNCH_SWT_CPU_ARCH_INDEX 0 CACHE STRING "CPU architecture index - fallback" FORCE)
        set(BNCH_SWT_CPU_ALIGNMENT 16 CACHE STRING "CPU Alignment" FORCE)
        set(BNCH_SWT_SIMD_FLAGS "" CACHE STRING "SIMD flags" FORCE)
        set(BNCH_SWT_SIMD_DEFINITIONS BNCH_SWT_SVE2=0;BNCH_SWT_AVX512=0;BNCH_SWT_AVX2=0;BNCH_SWT_NEON=0 CACHE STRING "SIMD definitions" FORCE)
        set(BNCH_SWT_INSTRUCTION_SET_NAME NONE CACHE STRING "Instruction set name" FORCE)
    endif()
endif()

message(STATUS "CPU Configuration: ${BNCH_SWT_THREAD_COUNT} threads, L1: ${BNCH_SWT_CPU_L1_CACHE_SIZE}B, arch index: ${BNCH_SWT_CPU_ARCH_INDEX}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/benchmarksuite_cpu_properties.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bnch_swt/benchmarksuite_cpu_properties.hpp
    @ONLY
)