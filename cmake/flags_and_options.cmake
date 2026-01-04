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
    BNCH_SWT_DEV=$<IF:$<STREQUAL:${BNCH_SWT_DEV},TRUE>,1,0>
    BNCH_SWT_CUDA_TENSOR_CORES=$<IF:$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<VERSION_GREATER_EQUAL:${CMAKE_CUDA_COMPILER_VERSION},11.0>>,1,0>
    BNCH_SWT_CUDA_MAX_REGISTERS=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,128,0>
    "BNCH_SWT_HOST_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__ __device__,__noinline__ __host__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_HOST=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__,__noinline__ __host__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_STATIC_HOST=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,static __forceinline__ __host__,static __noinline__ __host__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] static inline,inline static __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]] static ,__attribute__((noinline))>>>"
    "BNCH_SWT_NOINLINE_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__noinline__ __device__,__noinline__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_NOINLINE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__noinline__,__noinline__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __device__,__noinline__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "BNCH_SWT_GLOBAL=__global__"
    "half=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,__half,uint16_t>"
    "half2=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,__half2,uint32_t>"
    "bf16_t=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,__nv_bfloat16,uint16_t>"
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;WIN32_LEAN_AND_MEAN>
    ${BNCH_SWT_SIMD_DEFINITIONS}
)

# Updated to include UBSan for Clang and GCC
# Note: Sanitizers usually require -fno-omit-frame-pointer for clear stack traces

set(BNCH_SWT_UBSAN_FLAGS
    -fsanitize=undefined
    -fsanitize=float-divide-by-zero
    -fsanitize=float-cast-overflow
    -fno-sanitize-recover=all
    -fno-omit-frame-pointer
)

set(BNCH_SWT_CLANG_COMPILE_OPTIONS
    -O3
    ${BNCH_SWT_UBSAN_FLAGS}
    -funroll-loops
    -fvectorize
    -fslp-vectorize
    -finline-functions
    # -fomit-frame-pointer # Removed to allow UBSan traces
    -fmerge-all-constants
    -ffunction-sections
    -fdata-sections
    -falign-functions=32
    -fno-math-errno
    -ffp-contract=on
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fno-asynchronous-unwind-tables
    -fno-unwind-tables
    -fno-ident
    -pipe
    -fno-common
    -fwrapv
    -Weverything
    -Wnon-virtual-dtor
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-unsafe-buffer-usage
    -Wno-padded
    -Wno-c++20-compat
    -Wno-exit-time-destructors
    -Wno-c++20-extensions
)

set(BNCH_SWT_APPLECLANG_COMPILE_OPTIONS 
    -O3
    ${BNCH_SWT_UBSAN_FLAGS}
    -funroll-loops
    -fvectorize
    -fslp-vectorize
    -finline-functions
    # -fomit-frame-pointer # Removed to allow UBSan traces
    -fmerge-all-constants
    -ffunction-sections
    -fdata-sections
    -falign-functions=32
    -fno-math-errno
    -ffp-contract=on
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fno-asynchronous-unwind-tables
    -fno-unwind-tables
    -fno-ident
    -pipe
    -fno-common
    -fwrapv
    -Weverything
    -Wnon-virtual-dtor
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-unsafe-buffer-usage
    -Wno-padded
    -Wno-c++20-compat
    -Wno-exit-time-destructors
    -Wno-poison-system-directories
    -Wno-c++20-extensions
)

set(BNCH_SWT_GNU_COMPILE_OPTIONS 
    -O3
    ${BNCH_SWT_UBSAN_FLAGS}
    -funroll-loops
    -finline-functions
    # -fomit-frame-pointer # Removed to allow UBSan traces
    -fno-math-errno
    -falign-functions=32
    -falign-loops=32
    -fprefetch-loop-arrays
    -ftree-vectorize
    -fstrict-aliasing
    -ffunction-sections
    -fdata-sections
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fno-keep-inline-functions
    -fno-ident
    -fmerge-all-constants
    -fgcse-after-reload
    -ftree-loop-distribute-patterns
    -fpredictive-commoning
    -funswitch-loops
    -ftree-loop-vectorize
    -ftree-slp-vectorize
    -Wall
    -Wextra
    -Wpedantic
    -Wnon-virtual-dtor
    -Wlogical-op
    -Wduplicated-cond
    -Wduplicated-branches
    -Wnull-dereference
    -Wdouble-promotion
)

# You must also link against the UBSan runtime
set(BNCH_SWT_LINK_OPTIONS
    # Let the compiler driver handle the sanitizer runtime injection
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-fsanitize=undefined>
    
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
        -Wl,-x
        -Wl,-S
    >
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
        -Wl,-x
        -Wl,-S
    >
    # Remove -static-libubsan and let GCC try to use the dynamic runtime
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
    >
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:
        -Wl,--gc-sections
        -Wl,--strip-all
        -Wl,--build-id=none
        -Wl,--hash-style=gnu
        -Wl,-z,now
        -Wl,-z,relro
        -flto=thin
        -fwhole-program-vtables
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:
        -Wl,--gc-sections
        -Wl,--strip-all
        -Wl,--as-needed
        -Wl,-O3
    >
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:
        /DYNAMICBASE:NO
        /OPT:REF
        /OPT:ICF
        /INCREMENTAL:NO
        /MACHINE:X64
        /LTCG
    >
)