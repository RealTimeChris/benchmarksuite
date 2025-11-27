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

project(feature_detector
    VERSION 1.0.0
    LANGUAGES CXX
)

add_executable(feature_detector main.cpp)

include(${CMAKE_SOURCE_DIR}/../flags_and_options.cmake)

if(BNCH_SWT_DETECT_CPU_PROPERTIES)
    target_compile_definitions(feature_detector
        PUBLIC
            BNCH_SWT_DETECT_CPU_PROPERTIES
            ${BNCH_SWT_COMPILE_DEFINITIONS}
    )
    
elseif(BNCH_SWT_DETECT_GPU_PROPERTIES)
    find_package(CUDAToolkit REQUIRED)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES native)

    target_link_libraries(feature_detector
        PUBLIC
            CUDA::cudart
    )
    
    target_compile_definitions(feature_detector
        PUBLIC
            BNCH_SWT_DETECT_GPU_PROPERTIES
            ${BNCH_SWT_COMPILE_DEFINITIONS}
    )
endif()