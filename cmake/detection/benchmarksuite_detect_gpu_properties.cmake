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
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_gpu_properties.sh "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Gpu-Properties -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\" -DBNCH_SWT_DETECT_GPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Gpu-Properties --config=Release"
    )
    
    execute_process(
        COMMAND chmod +x ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_gpu_properties.sh
        RESULT_VARIABLE CHMOD_RESULT
    )
    
    if(NOT CHMOD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for build_feature_tester_gpu_properties.sh")
    endif()
    
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_gpu_properties.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection
    )
    
    set(FEATURE_TESTER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/Build-Gpu-Properties/feature_detector)
    
elseif(WIN32)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_gpu_properties.bat
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Gpu-Properties -DBNCH_SWT_DETECT_GPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Gpu-Properties --config=Release"
    )
    
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/build_feature_tester_gpu_properties.bat
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection
    )
    
    set(FEATURE_TESTER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/Build-Gpu-Properties/Release/feature_detector.exe)
endif()

if(NOT DEFINED BNCH_SWT_SM_COUNT OR
   NOT DEFINED BNCH_SWT_MAX_THREADS_PER_SM OR
   NOT DEFINED BNCH_SWT_MAX_THREADS_PER_BLOCK OR
   NOT DEFINED BNCH_SWT_WARP_SIZE OR
   NOT DEFINED BNCH_SWT_GPU_L2_CACHE_SIZE OR
   NOT DEFINED BNCH_SWT_SHARED_MEM_PER_BLOCK OR
   NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_X OR
   NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Y OR
   NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Z OR
   NOT DEFINED BNCH_SWT_GPU_ARCH_INDEX OR
   NOT BNCH_SWT_DETECT_GPU_PROPERTIES)
    
    execute_process(
        COMMAND ${FEATURE_TESTER_FILE}
        RESULT_VARIABLE FEATURE_TESTER_EXIT_CODE
        OUTPUT_VARIABLE GPU_PROPERTIES_OUTPUT
        ERROR_VARIABLE FEATURE_TESTER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(FEATURE_TESTER_EXIT_CODE EQUAL 0 AND GPU_PROPERTIES_OUTPUT MATCHES "GPU_SUCCESS=1")
    
    string(REGEX MATCH "SM_COUNT=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_SM_COUNT)
        set(BNCH_SWT_SM_COUNT ${CMAKE_MATCH_1} CACHE STRING "GPU SM count" FORCE)
    endif()
    
    string(REGEX MATCH "MAX_THREADS_PER_SM=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAX_THREADS_PER_SM)
        set(BNCH_SWT_MAX_THREADS_PER_SM ${CMAKE_MATCH_1} CACHE STRING "GPU max threads per SM" FORCE)
    endif()
    
    string(REGEX MATCH "MAX_THREADS_PER_BLOCK=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAX_THREADS_PER_BLOCK)
        set(BNCH_SWT_MAX_THREADS_PER_BLOCK ${CMAKE_MATCH_1} CACHE STRING "GPU max threads per block" FORCE)
    endif()
    
    string(REGEX MATCH "WARP_SIZE=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_WARP_SIZE)
        set(BNCH_SWT_WARP_SIZE ${CMAKE_MATCH_1} CACHE STRING "GPU warp size" FORCE)
    endif()
    
    string(REGEX MATCH "L2_CACHE_SIZE=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_GPU_L2_CACHE_SIZE)
        set(BNCH_SWT_GPU_L2_CACHE_SIZE ${CMAKE_MATCH_1} CACHE STRING "GPU L2 cache size" FORCE)
    endif()
    
    string(REGEX MATCH "SHARED_MEM_PER_BLOCK=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_SHARED_MEM_PER_BLOCK)
        set(BNCH_SWT_SHARED_MEM_PER_BLOCK ${CMAKE_MATCH_1} CACHE STRING "GPU shared memory per block" FORCE)
    endif()
    
    string(REGEX MATCH "MAX_GRID_SIZE_X=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_X)
        set(BNCH_SWT_MAX_GRID_SIZE_X ${CMAKE_MATCH_1} CACHE STRING "GPU max grid size X" FORCE)
    endif()

    string(REGEX MATCH "MAX_GRID_SIZE_Y=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Y)
        set(BNCH_SWT_MAX_GRID_SIZE_Y ${CMAKE_MATCH_1} CACHE STRING "GPU max grid size Y" FORCE)
    endif()

    string(REGEX MATCH "MAX_GRID_SIZE_Z=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Z)
        set(BNCH_SWT_MAX_GRID_SIZE_Z ${CMAKE_MATCH_1} CACHE STRING "GPU max grid size Z" FORCE)
    endif()

    string(REGEX MATCH "MAJOR_COMPUTE_CAPABILITY=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MAJOR_COMPUTE_CAPABILITY)
        set(BNCH_SWT_MAJOR_COMPUTE_CAPABILITY ${CMAKE_MATCH_1} CACHE STRING "GPU major compute capability" FORCE)
    endif()
    
    string(REGEX MATCH "MINOR_COMPUTE_CAPABILITY=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_MINOR_COMPUTE_CAPABILITY)
        set(BNCH_SWT_MINOR_COMPUTE_CAPABILITY ${CMAKE_MATCH_1} CACHE STRING "GPU minor compute capability" FORCE)
    endif()
    
    string(REGEX MATCH "GPU_ARCH_INDEX=([0-9]+)" _ ${GPU_PROPERTIES_OUTPUT})
    if(NOT DEFINED BNCH_SWT_GPU_ARCH_INDEX)
        set(BNCH_SWT_GPU_ARCH_INDEX ${CMAKE_MATCH_1} CACHE STRING "GPU architecture index" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_GPU_PROPERTIES_ERECTED)
        set(BNCH_SWT_GPU_PROPERTIES_ERECTED TRUE CACHE BOOL "GPU properties successfully detected" FORCE)
    endif()
    
    message(STATUS "GPU Properties detected successfully")
    
else()
    message(WARNING "GPU feature detector failed, using reasonable default values for unset properties")
    
    if(NOT DEFINED BNCH_SWT_SM_COUNT)
        set(BNCH_SWT_SM_COUNT 16 CACHE STRING "GPU SM count (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_MAX_THREADS_PER_SM)
        set(BNCH_SWT_MAX_THREADS_PER_SM 1024 CACHE STRING "GPU max threads per SM (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_MAX_THREADS_PER_BLOCK)
        set(BNCH_SWT_MAX_THREADS_PER_BLOCK 1024 CACHE STRING "GPU max threads per block (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_WARP_SIZE)
        set(BNCH_SWT_WARP_SIZE 32 CACHE STRING "GPU warp size (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_GPU_L2_CACHE_SIZE)
        set(BNCH_SWT_GPU_L2_CACHE_SIZE 2097152 CACHE STRING "GPU L2 cache size (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_SHARED_MEM_PER_BLOCK)
        set(BNCH_SWT_SHARED_MEM_PER_BLOCK 49152 CACHE STRING "GPU shared memory per block (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_X)
        set(BNCH_SWT_MAX_GRID_SIZE_X 2147483647 CACHE STRING "GPU max grid size X (fallback)" FORCE)
    endif()

    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Y)
        set(BNCH_SWT_MAX_GRID_SIZE_Y 65535 CACHE STRING "GPU max grid size Y (fallback)" FORCE)
    endif()

    if(NOT DEFINED BNCH_SWT_MAX_GRID_SIZE_Z)
        set(BNCH_SWT_MAX_GRID_SIZE_Z 65535 CACHE STRING "GPU max grid size Z (fallback)" FORCE)
    endif()
    
    if(NOT DEFINED BNCH_SWT_GPU_ARCH_INDEX)
        set(BNCH_SWT_GPU_ARCH_INDEX 0 CACHE STRING "GPU architecture index (fallback)" FORCE)
    endif()
endif()

if(NOT DEFINED BNCH_SWT_TOTAL_THREADS)
    math(EXPR BNCH_SWT_TOTAL_THREADS "${BNCH_SWT_SM_COUNT} * ${BNCH_SWT_MAX_THREADS_PER_SM}")
    set(BNCH_SWT_TOTAL_THREADS ${BNCH_SWT_TOTAL_THREADS} CACHE STRING "GPU total concurrent threads" FORCE)
endif()

message(STATUS "GPU Configuration: ${BNCH_SWT_SM_COUNT} SMs, ${BNCH_SWT_TOTAL_THREADS} total threads")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detection/benchmarksuite_gpu_properties.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bnch_swt/benchmarksuite_gpu_properties.hpp
    @ONLY
)