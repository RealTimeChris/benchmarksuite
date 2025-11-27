/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/benchmarksuite
#if defined(BNCH_SWT_DETECT_GPU_PROPERTIES)
	#include <cuda_runtime.h>
	#include <iostream>

int32_t main() {
	cudaDeviceProp device_prop;
	cudaError_t result = cudaGetDeviceProperties(&device_prop, 0);

	if (result != cudaSuccess) {
		std::cout << "CUDA_ERROR=1" << std::endl;
		return 1;
	}

	uint32_t gpu_arch_index = 0;
	if (device_prop.major == 9) {
		gpu_arch_index = 1;
	} else if (device_prop.major == 10) {
		gpu_arch_index = 2;
	} else if (device_prop.major == 11) {
		gpu_arch_index = 3;
	} else if (device_prop.major == 12) {
		gpu_arch_index = 4;
	} else {
		gpu_arch_index = 0;
	}

	std::cout << "SM_COUNT=" << device_prop.multiProcessorCount << std::endl;
	std::cout << "MAX_THREADS_PER_SM=" << device_prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "MAX_THREADS_PER_BLOCK=" << device_prop.maxThreadsPerBlock << std::endl;
	std::cout << "WARP_SIZE=" << device_prop.warpSize << std::endl;
	std::cout << "L2_CACHE_SIZE=" << device_prop.l2CacheSize << std::endl;
	std::cout << "SHARED_MEM_PER_BLOCK=" << device_prop.sharedMemPerBlock << std::endl;
	std::cout << "MEMORY_BUS_WIDTH=" << device_prop.memoryBusWidth << std::endl;
	std::cout << "MEMORY_CLOCK_RATE=" << device_prop.memoryClockRate << std::endl;
	std::cout << "MAJOR_COMPUTE_CAPABILITY=" << device_prop.major << std::endl;
	std::cout << "MINOR_COMPUTE_CAPABILITY=" << device_prop.minor << std::endl;
	std::cout << "MAX_GRID_SIZE_X=" << device_prop.maxGridSize[0] << std::endl;
	std::cout << "MAX_GRID_SIZE_Y=" << device_prop.maxGridSize[1] << std::endl;
	std::cout << "MAX_GRID_SIZE_Z=" << device_prop.maxGridSize[2] << std::endl;
	std::cout << "MAX_BLOCK_SIZE_X=" << device_prop.maxThreadsPerBlock << std::endl;
	std::cout << "GPU_ARCH_INDEX=" << gpu_arch_index << std::endl;
	std::cout << "GPU_SUCCESS=1" << std::endl;

	return 0;
}
#elif defined(BNCH_SWT_DETECT_CPU_PROPERTIES)
	#include <cstring>
	#include <cstdint>
	#include <cstdlib>
	#include <iostream>
	#include <thread>
	#include <vector>

	#if BNCH_SWT_COMPILER_MSVC
		#include <intrin.h>
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
		#include <cpuid.h>
	#endif

	#if BNCH_SWT_PLATFORM_WINDOWS
		#include <Windows.h>
	#endif
	#if BNCH_SWT_PLATFORM_LINUX || BNCH_SWT_PLATFORM_ANDROID
		#include <fstream>
		#include <string>
	#endif
	#if BNCH_SWT_PLATFORM_MAC
		#include <sys/sysctl.h>
		#include <sys/types.h>
		#include <string>
	#endif

	#if BNCH_SWT_ARCH_ARM64
		#if BNCH_SWT_PLATFORM_LINUX
			#include <sys/auxv.h>
			#include <asm/hwcap.h>
		#elif BNCH_SWT_PLATFORM_MAC
			#include <sys/sysctl.h>
		#endif
	#endif

enum class instruction_set {
	FALLBACK = 0x0,
	AVX2	 = 0x1,
	AVX512f	 = 0x2,
	NEON	 = 0x4,
	SVE2	 = 0x8,
};

enum class cache_level {
	one	  = 1,
	two	  = 2,
	three = 3,
};

	#if BNCH_SWT_ARCH_ARM64
inline static uint32_t detect_supported_architectures() {
	uint32_t host_isa = static_cast<uint32_t>(instruction_set::NEON);

		#if BNCH_SWT_PLATFORM_LINUX
	unsigned long hwcap = getauxval(AT_HWCAP);
	if (hwcap & HWCAP_SVE) {
		host_isa |= static_cast<uint32_t>(instruction_set::SVE2);
	}
		#endif

	return host_isa;
}

	#elif BNCH_SWT_ARCH_X64
static constexpr uint32_t cpuid_avx2_bit	 = 1ul << 5;
static constexpr uint32_t cpuid_avx512_bit	 = 1ul << 16;
static constexpr uint64_t cpuid_avx256_saved = 1ull << 2;
static constexpr uint64_t cpuid_avx512_saved = 7ull << 5;
static constexpr uint32_t cpuid_osx_save	 = (1ul << 26) | (1ul << 27);

inline static void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
		#if BNCH_SWT_COMPILER_MSVC
	int32_t cpu_info[4];
	__cpuidex(cpu_info, *eax, *ecx);
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
		#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	uint32_t level = *eax;
	__get_cpuid(level, eax, ebx, ecx, edx);
		#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "c"(c));
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
		#endif
}

inline static uint64_t xgetbv() {
		#if BNCH_SWT_COMPILER_MSVC
	return _xgetbv(0);
		#else
	uint32_t eax, edx;
	asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (( uint64_t )edx << 32) | eax;
		#endif
}

inline static uint32_t detect_supported_architectures() {
	std::uint32_t eax	   = 0;
	std::uint32_t ebx	   = 0;
	std::uint32_t ecx	   = 0;
	std::uint32_t edx	   = 0;
	std::uint32_t host_isa = static_cast<uint32_t>(instruction_set::FALLBACK);

	eax = 0x1;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if ((ecx & cpuid_osx_save) != cpuid_osx_save) {
		return host_isa;
	}

	uint64_t xcr0 = xgetbv();
	if ((xcr0 & cpuid_avx256_saved) == 0) {
		return host_isa;
	}

	eax = 0x7;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if (ebx & cpuid_avx2_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
	}

	if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
		return host_isa;
	}

	if (ebx & cpuid_avx512_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX512f);
	}

	return host_isa;
}

	#else
inline static uint32_t detect_supported_architectures() {
	return static_cast<uint32_t>(instruction_set::FALLBACK);
}
	#endif

inline uint64_t get_cache_size(cache_level level) {
	#if BNCH_SWT_PLATFORM_WINDOWS
	DWORD buffer_size = 0;
	std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};
	GetLogicalProcessorInformation(nullptr, &buffer_size);
	buffer.resize(buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

	if (!GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
		return 0;
	}

	const auto info_count = buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
	for (uint64_t i = 0; i < info_count; ++i) {
		if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == static_cast<int32_t>(level)) {
			if (level == cache_level::one && buffer[i].Cache.Type == CacheData) {
				return buffer[i].Cache.Size;
			} else if (level != cache_level::one && buffer[i].Cache.Type == CacheUnified) {
				return buffer[i].Cache.Size;
			}
		}
	}
	return 0;

	#elif BNCH_SWT_PLATFORM_LINUX || BNCH_SWT_PLATFORM_ANDROID
	auto get_cache_size_from_file = [](const std::string& index) {
		const std::string cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index" + index + "/size";
		std::ifstream file(cache_file_path);
		if (!file.is_open()) {
			return static_cast<uint64_t>(0);
		}

		std::string size_str;
		file >> size_str;
		file.close();

		uint64_t size = 0;
		if (size_str.back() == 'K') {
			size = std::stoul(size_str) * 1024;
		} else if (size_str.back() == 'M') {
			size = std::stoul(size_str) * 1024 * 1024;
		} else {
			size = std::stoul(size_str);
		}
		return size;
	};

	if (level == cache_level::one) {
		return get_cache_size_from_file("0");
	} else {
		std::string index = (level == cache_level::two) ? "2" : "3";
		return get_cache_size_from_file(index);
	}

	#elif BNCH_SWT_PLATFORM_MAC
	auto get_cache_size_for_mac = [](const std::string& cache_type) {
		uint64_t cache_size		 = 0;
		size_t size			 = sizeof(cache_size);
		std::string sysctl_query = "hw." + cache_type + "cachesize";
		if (sysctlbyname(sysctl_query.c_str(), &cache_size, &size, nullptr, 0) != 0) {
			return uint64_t{ 0 };
		}
		return cache_size;
	};

	if (level == cache_level::one) {
		return get_cache_size_for_mac("l1d");
	} else if (level == cache_level::two) {
		return get_cache_size_for_mac("l2");
	} else {
		return get_cache_size_for_mac("l3");
	}
	#endif

	return 0;
}

int32_t main() {
	const uint32_t thread_count	 = std::thread::hardware_concurrency();
	const uint32_t supported_isa = detect_supported_architectures();
	const uint64_t l1_cache_size = get_cache_size(cache_level::one);
	const uint64_t l2_cache_size = get_cache_size(cache_level::two);
	const uint64_t l3_cache_size = get_cache_size(cache_level::three);
	std::cout << "THREAD_COUNT=" << thread_count << std::endl;
	std::cout << "INSTRUCTION_SET=" << supported_isa << std::endl;
	std::cout << "HAS_AVX2=" << ((supported_isa & static_cast<uint32_t>(instruction_set::AVX2)) ? 1 : 0) << std::endl;
	std::cout << "HAS_AVX512=" << ((supported_isa & static_cast<uint32_t>(instruction_set::AVX512f)) ? 1 : 0) << std::endl;
	std::cout << "HAS_NEON=" << ((supported_isa & static_cast<uint32_t>(instruction_set::NEON)) ? 1 : 0) << std::endl;
	std::cout << "HAS_SVE2=" << ((supported_isa & static_cast<uint32_t>(instruction_set::SVE2)) ? 1 : 0) << std::endl;
	std::cout << "L1_CACHE_SIZE=" << l1_cache_size << std::endl;
	std::cout << "L2_CACHE_SIZE=" << l2_cache_size << std::endl;
	std::cout << "L3_CACHE_SIZE=" << l3_cache_size << std::endl;
	std::cout << "CPU_SUCCESS=1" << std::endl;
	return 0;
}
#else
int32_t main() {
	return -1;
}
#endif