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

if(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows-static CACHE STRING "" FORCE)
elseif(UNIX AND NOT APPLE)
    set(VCPKG_TARGET_TRIPLET x64-linux CACHE STRING "" FORCE)
elseif(APPLE)
    set(VCPKG_TARGET_TRIPLET x64-osx CACHE STRING "" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(BNCH_SWT_DETECT_CPU_PROPERTIES "Override cpu-cache-size selection" OFF)
option(BNCH_SWT_DETECT_GPU_PROPERTIES "Override cuda-cache-size selection" OFF)
option(BNCH_SWT_NO_INSTALL "Install nihilus as a library" OFF)
option(BNCH_SWT_CUDA "Enable CUDA support" OFF)
option(BNCH_SWT_DEV "Development mode" OFF)
option(BNCH_SWT_DEPLOY "Deployment mode" OFF)
option(BNCH_SWT_CPU "CPU build" OFF)

if(BNCH_SWT_CUDA)
    set(BNCH_SWT_LANGUAGE CUDA)
    project(benchmarksuite
        VERSION 1.0.0
        DESCRIPTION "benchmarksuite: You trained the model. benchmarksuite gives it a voice."
        LANGUAGES CUDA 
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
        VERSION 1.0.0
        DESCRIPTION "benchmarksuite: You trained the model. benchmarksuite gives it a voice."
        LANGUAGES CXX
    )
    set(BNCH_SWT_MAIN_FILE_EXTENSION .cpp)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/detection/benchmarksuite_detect_cpu_properties.cmake)

set(BNCH_SWT_COMPILER_ID ${CMAKE_${BNCH_SWT_LANGUAGE}_COMPILER_ID})

file(GLOB_RECURSE BNCH_SWT_HEADERS CONFIGURE_DEPENDS include/*.hpp)

add_library(${PROJECT_NAME} INTERFACE ${BNCH_SWT_HEADERS})

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(cmake/flags_and_options.cmake)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        $<$<TARGET_EXISTS:CUDA::cudart_static>:CUDA::cudart_static>
)

target_compile_options(${PROJECT_NAME}
    INTERFACE
        ${BNCH_SWT_COMPILE_OPTIONS}
)

target_link_options(${PROJECT_NAME}
    INTERFACE
        ${BNCH_SWT_LINK_OPTIONS}
)

target_compile_definitions(${PROJECT_NAME}
    INTERFACE
        ${BNCH_SWT_COMPILE_DEFINITIONS}
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/benchmarksuite/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	INSTALL_DESTINATION "share/benchmarksuite"
	PATH_VARS
	EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	VERSION "${PRODUCT_VERSION}"
	COMPATIBILITY AnyNewerVersion
)

if (NOT BNCH_SWT_NO_INSTALL)
    install(
	    FILES
	    "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	    "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	    DESTINATION "share/benchmarksuite"
    )

    install(
	    DIRECTORY
	    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
	    DESTINATION "include"
    )

    install(
	    TARGETS "${PROJECT_NAME}"
	    EXPORT "${EXPORTED_TARGETS_NAME}"
    )

    install(
	    EXPORT "${EXPORTED_TARGETS_NAME}"
	    FILE "${EXPORTED_TARGETS_FILE_NAME}"
	    NAMESPACE "${PROJECT_NAME}::"
	    DESTINATION "share/benchmarksuite"
    )
endif()

message(STATUS "benchmarksuite Configuration Summary:")
message(STATUS "=============================")
message(STATUS "Version: ${PROJECT_VERSION}")

if(BNCH_SWT_CUDA)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(BENCHMARKS)
    add_subdirectory(src)
endif()