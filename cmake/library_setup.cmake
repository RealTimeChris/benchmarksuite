# library_setup.cmake - Script for detecting the CPU architecture.
# MIT License
# Copyright (c) 2026 RealTimeChris

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set(BNCH_SWT_COMPILE_DEFINITIONS
    BNCH_SWT_COMPILER_CUDA=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,1,0>
    BNCH_SWT_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>
    BNCH_SWT_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>
    BNCH_SWT_PLATFORM_ANDROID=$<IF:$<PLATFORM_ID:Android>,1,0>
    BNCH_SWT_PLATFORM_WINDOWS=$<IF:$<PLATFORM_ID:Windows>,1,0>
    BNCH_SWT_PLATFORM_LINUX=$<IF:$<PLATFORM_ID:Linux>,1,0>
    BNCH_SWT_PLATFORM_MAC=$<IF:$<PLATFORM_ID:Darwin>,1,0>
    BNCH_SWT_COMPILER_CLANG=$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,1,0>
    BNCH_SWT_COMPILER_MSVC=$<IF:$<CXX_COMPILER_ID:MSVC>,1,0>
    BNCH_SWT_COMPILER_GCC=$<IF:$<CXX_COMPILER_ID:GNU>,1,0>
    BNCH_SWT_CPB_ASCII=$<IF:$<CXX_COMPILER_ID:MSVC>,8,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,16,$<IF:$<CXX_COMPILER_ID:GNU>,1,8>>>
    BNCH_SWT_BPS_ASCII=$<IF:$<CXX_COMPILER_ID:MSVC>,2,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,2,$<IF:$<CXX_COMPILER_ID:GNU>,4,4>>>
    BNCH_SWT_CPB_MIXED=$<IF:$<CXX_COMPILER_ID:MSVC>,8,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,4,$<IF:$<CXX_COMPILER_ID:GNU>,1,4>>>
    BNCH_SWT_BPS_MIXED=$<IF:$<CXX_COMPILER_ID:MSVC>,8,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,1,$<IF:$<CXX_COMPILER_ID:GNU>,2,1>>>
    BNCH_SWT_CPB_MULTIBYTE=$<IF:$<CXX_COMPILER_ID:MSVC>,8,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,8,$<IF:$<CXX_COMPILER_ID:GNU>,1,4>>>
    BNCH_SWT_BPS_MULTIBYTE=$<IF:$<CXX_COMPILER_ID:MSVC>,1,$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,1,$<IF:$<CXX_COMPILER_ID:GNU>,1,1>>>
    "BNCH_SWT_HOST_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__ __device__,__noinline__ __host__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_HOST=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__,__noinline__ __host__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_STATIC_HOST=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,static __forceinline__ __host__,static __noinline__ __host__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] static inline,inline static __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]] static ,__attribute__((noinline))>>>"
    "BNCH_SWT_NOINLINE_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__noinline__ __device__,__noinline__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_NOINLINE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__noinline__,__noinline__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __device__,__noinline__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_LIFETIME_BOUND=$<IF:$<CXX_COMPILER_ID:Clang>,[[clang::lifetimebound]],>"
    "BNCH_SWT_GLOBAL=__global__"
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;WIN32_LEAN_AND_MEAN>
)

set(BNCH_SWT_COMPILER_ID ${CMAKE_${BNCH_SWT_LANGUAGE}_COMPILER_ID})

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(${PROJECT_NAME}
    INTERFACE
    ${BNCH_SWT_SIMD_FLAGS}
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        $<$<TARGET_EXISTS:CUDA::cudart_static>:CUDA::cudart_static>
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_VERSION VERSION_GREATER "24.99.99")
    set(BNCH_SWT_DISABLE_PINNING 1)
else()
    set(BNCH_SWT_DISABLE_PINNING 0)
endif()

target_compile_definitions(${PROJECT_NAME}
    INTERFACE
	BNCH_SWT_OPERATING_SYSTEM_VERSION=\"${CMAKE_SYSTEM_VERSION}\"
	BNCH_SWT_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\"
    BNCH_SWT_CPU_SUPPORT=$<IF:$<BOOL:${BNCH_SWT_BUILD_CPU_SUPPORT}>,1,0>
    BNCH_SWT_GPU_SUPPORT=$<IF:$<BOOL:${BNCH_SWT_BUILD_GPU_SUPPORT}>,1,0>
    BNCH_SWT_DISABLE_PINNING=${BNCH_SWT_DISABLE_PINNING}
    BNCH_SWT_CHUNKS_PER_BLOCK=4
    BNCH_SWT_BLOCKS_PER_STEP=8
    ${BNCH_SWT_SIMD_DEFINITIONS}
    ${BNCH_SWT_COMPILE_DEFINITIONS}
)
